# RMA-Style Environment Encoder

This document describes the MVP implementation for estimating road friction
`mu` from recent closed-loop state/action history.

## Goal

The current PPO+LPV-MPC baseline passes the true road friction `env.mu` directly
into the PPO actor/critic input. The RMA-style encoder removes that direct
runtime dependency in stages:

1. Phase 1: train `history -> mu_hat` with supervised labels from `env.mu`.
2. Phase 2: replace the PPO input feature `env.mu` with frozen encoder output
   `mu_hat`.
3. Phase 3: replace scalar `mu_hat` with a latent vector `z` for multiple
   environment factors.

The implemented MVP covers Phase 1 and the Phase 2 runtime switch.

## Implemented Files

- `src/utils/history_buffer.py`
  - Defines the fixed history feature order.
  - Builds one normalized history row from `obs`, `action_mpc`, and PPO `alpha`.
  - Maintains a fixed-length padded `HistoryBuffer`.

- `src/agents/mlp_env_encoder.py`
  - Defines `MLPEnvEncoder`, an MLP-only encoder with no CNN/RNN/Transformer.
  - Predicts `mu_hat` in `[0.2, 1.0]` using a sigmoid output mapping.
  - Defines `EncoderDataset` for `.npz` files collected from rollouts.

- `src/agents/env_encoder.py`
  - Backward-compatible alias: `EnvEncoder = MLPEnvEncoder`.
  - Prevents accidental reuse of the old GRU implementation.

- `src/utils/utils.py`
  - Adds `build_policy_state(obs, env_feature)`.
  - This centralizes PPO input construction as `Normalize(obs[:3]) + env_feature`.

- `scripts/collect_encoder_data.py`
  - Runs closed-loop rollouts and saves supervised encoder samples.
  - Control still uses the baseline true `env.mu` PPO input during collection.

- `scripts/train_env_encoder.py`
  - Trains the MLP encoder with MSE loss.
  - Saves a checkpoint containing model weights and encoder configuration.

- `scripts/eval_env_encoder.py`
  - Evaluates a saved encoder checkpoint.
  - Reports overall MSE/MAE and low/mid/high `mu` segment MAE.

- `scripts/main.py`
  - Adds `--mu_source true|encoder`.
  - Default is `--mu_source true`, preserving the previous baseline behavior.
  - `--mu_source encoder` loads a frozen encoder checkpoint and feeds `mu_hat`
    to PPO instead of true `env.mu`.

## History Feature Format

Each timestep row has 9 normalized features:

```text
[vx, vy, r, y, psi, action_mpc, alpha_q1, alpha_q2, alpha_r]
```

The default window is 25 control steps, so each encoder sample has shape:

```text
(25, 9)
```

Rows are normalized with fixed scales:

```text
[70.0, 0.4, 4.0, 10.0, 10.0, 1000.0, 2.0, 2.0, 2.0]
```

Short histories at the start of an episode are padded by repeating the first
available row. Histories never cross episode boundaries because each episode
creates a new `HistoryBuffer`.

## Data Format

`collect_encoder_data.py` writes a compressed `.npz` file with:

```text
histories:   float32 array, shape (N, history_len, 9)
labels:      float32 array, shape (N,), true env.mu
episode_ids: int32 array, shape (N,)
times:       float32 array, shape (N,)
feature_names
history_len
```

Only `histories` and `labels` are required by `EncoderDataset`.

## Recommended Workflow

Run these commands from the project root.

### 1. Collect supervised data

Use the trained PPO checkpoint if available:

```powershell
python scripts\collect_encoder_data.py `
  --episodes 50 `
  --use_checkpoint `
  --ppo_checkpoint ppo_checkpoint_episode2.pth `
  --output runs\encoder_data\encoder_dataset.npz
```

For a quick pipeline check, reduce `--episodes`.

### 2. Train the encoder

```powershell
python scripts\train_env_encoder.py `
  --data runs\encoder_data\encoder_dataset.npz `
  --output runs\encoder_checkpoints\mlp_env_encoder.pth `
  --epochs 50 `
  --batch_size 256
```

### 3. Evaluate offline

```powershell
python scripts\eval_env_encoder.py `
  --data runs\encoder_data\encoder_dataset.npz `
  --checkpoint runs\encoder_checkpoints\mlp_env_encoder.pth
```

Inspect:

- overall `mae`
- `mae_mu_lt_0.4`
- `mae_mu_0.4_to_0.7`
- `mae_mu_gt_0.7`

Low-friction error is the most important segment for stability control.

### 4. Test PPO with estimated `mu_hat`

```powershell
python scripts\main.py `
  --mode test `
  --mu_source encoder `
  --encoder_checkpoint runs\encoder_checkpoints\mlp_env_encoder.pth
```

To run the unchanged baseline:

```powershell
python scripts\main.py --mode test --mu_source true
```

## Notes For Experiments

- I did not run CarSim rollouts during implementation. The code was validated
  with syntax checks, CLI checks, and a synthetic train/eval smoke test.
- `env.mu` is still available in Phase 2 for logging and error measurement, but
  it is not fed into PPO when `--mu_source encoder` is selected.
- The LPV-MPC model still does not consume `mu` directly. This is intentional for
  the MVP and keeps the change minimally invasive.
- If `mu_hat` causes performance regression, first evaluate the frozen PPO with
  `mu_hat`, then retrain PPO using `--mu_source encoder`.

## Extension To Latent `z`

For Phase 3, replace the scalar output head with a latent vector:

```text
Encoder(history) -> z
```

PPO then changes from:

```text
[Normalize(obs[:3]), mu_hat]
```

to:

```text
[Normalize(obs[:3]), z]
```

Optional supervised heads can still predict interpretable quantities such as
`mu`, tire burst severity, burst location, mass shift, or cornering stiffness
scale for monitoring and auxiliary losses.
