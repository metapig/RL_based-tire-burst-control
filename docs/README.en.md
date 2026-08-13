# Reinforcement-Learning-Based Vehicle Tire-Burst Stabilization

[中文](../README.md) | [English](README.en.md) | [Deutsch](README.de.md) | [한국어](README.ko.md)

This project trains a continuous-action PPO controller with CarSim and PyTorch to stabilize a vehicle and track its trajectory after a tire burst. Training randomizes initial speed, road friction, vehicle mass, burst wheel, and burst severity. The current design uses privileged environment information during policy training and provides a history encoder that estimates the same latent representation from recent observations.

> This repository does not include CarSim, its DLLs, `simfile.sim`, local database files, trained weights, or TensorBoard logs. A valid CarSim installation and local configuration are required for full simulation.

## Method

- **Algorithm:** PPO with GAE for five-dimensional continuous control.
- **Actions:** front steering and four wheel-control moments, scaled by `(3, 400, 400, 400, 400)`.
- **Bounded policy:** Gaussian latent actions are transformed by `tanh` and physical action scales. Pre-tanh actions are retained for consistent PPO log-probability evaluation.
- **Exploration:** one learnable global `log_std` per action dimension, constrained to a configured range.
- **Privileged training:** burst wheel/severity, road friction, and vehicle mass are encoded into a latent variable.
- **History adaptation:** a 1-D CNN predicts the privileged latent from 25 state-history steps for control without direct access to environment parameters.
- **Curriculum:** the ranges of speed, low friction, and burst severity expand with the episode index.

## Repository layout

```text
tire_burst/
├── generated_tire_tables/     # 25/50/75/100% Fx/Fy/Mz burst tables
├── scripts/                   # training, adaptation, and comparison entry points
├── src/
│   ├── agents/                # PPO and history encoder
│   ├── envs/                  # CarSim wrapper and tire-burst environment
│   ├── models/                # state, reward, and parameters
│   ├── mpc/                   # LPV-MPC reference implementation
│   └── utils/                 # VS Solver and tire-table tools
├── tests/
├── docs/
├── TRAINING_COMMANDS.md       # detailed PowerShell command guide (Chinese)
└── README.md                  # Chinese main README
```

Local runs additionally create `checkpoints/`, `datasets/`, `runs/`, and `simfile.sim`; all are excluded by `.gitignore`.

## Requirements

- Windows and Python 3.10+
- CarSim 2023; the current integration targets the CarSim 2023.2 interface
- PyTorch, NumPy, and TensorBoard
- CasADi for the LPV-MPC module
- `pefile`, `tomlkit`, and `pytest`

Example setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy torch tensorboard casadi pefile tomlkit pytest
```

Dependency versions are not locked yet. Record Python, PyTorch, CUDA, CarSim, and package versions for reproducible experiments.

## CarSim configuration

1. Put a valid `simfile.sim` in the repository root.
2. Replace development-machine absolute paths in:
   - `src/envs/tire_burst_env.py`
   - `src/utils/generate_burst_force_tables.py`
   - `src/utils/get_dll_functions.py`
3. Ensure the simulation exposes 6 import channels and at least 8 export variables in the expected order.
4. Back up the CarSim database: environment resets write the selected severity tables into configured tire datasets.

Paths stored in `generated_tire_tables/manifest.json` document data provenance and are not portable configuration values.

## Quick start

Run tests:

```powershell
python -m pytest -q
```

Train from scratch:

```powershell
python -m scripts.main --mode train --max_episodes 4000 --checkpoint_interval 100 --checkpoint "checkpoints\ppo_squashed.pth"
```

Resume for 1,000 additional episodes:

```powershell
python -m scripts.main --mode resume --max_episodes 1000 --checkpoint_interval 100 --checkpoint "checkpoints\ppo_squashed.pth"
```

Test the deterministic privileged/oracle policy:

```powershell
python -m scripts.main --mode test --checkpoint "checkpoints\ppo_squashed.pth"
```

See [TRAINING_COMMANDS.md](../TRAINING_COMMANDS.md) for data collection, history-encoder training, paired evaluation, TensorBoard, and tire-table commands.

## History adaptation

```powershell
python -m scripts.main --mode collect_history_data --checkpoint "checkpoints\ppo_squashed.pth" --adaptation_episodes 500 --dataset "datasets\adaptation.pt"
python -m scripts.train_history_encoder --train --dataset "datasets\adaptation.pt" --checkpoint "checkpoints\history_encoder.pth" --epochs 30 --batch_size 64 --learning_rate 0.0005
python -m scripts.compare_oracle_history --checkpoint "checkpoints\ppo_squashed.pth" --history_checkpoint "checkpoints\history_encoder.pth" --episodes 20
```

## Tire-table tools

```powershell
python -m src.utils.generate_burst_force_tables --levels 0.25 0.5 0.75 1.0 --output-dir "generated_tire_tables" --reference-dir "generated_tire_tables\burst_075" --reference-severity 0.75 --mz-full-burst-scale 0.25
python -m src.utils.restore_full_burst_tables --table-dir "generated_tire_tables\burst_100"
```

The restore command modifies local CarSim datasets. Verify paths and create a backup first.

## Reward and statistical interpretation

`src/models/reward_model.py` combines speed, tracking, stability, control magnitude, action smoothness, and abnormal-termination components. Changing the reward and resuming training defines a new optimization objective initialized by the old policy. Episode returns computed under different rewards are not directly comparable.

Use a fixed, independent, paired scenario set and compare physical outcomes: tracking error, yaw response, lateral velocity, action RMS/saturation, failure rate, and tail performance. The current entry point does not yet provide complete command-line seed control.

## Ignored artifacts

The repository excludes TensorBoard runs, checkpoints, generated datasets, `.sim/.par/.dll` CarSim artifacts, caches, virtual environments, logs, and editor metadata. Publish models through GitHub Releases, Git LFS, or an external model registry together with the exact commit, reward version, command, and evaluation results.

## Status and license

This is a research prototype, not production vehicle-control software. Real-vehicle testing requires independent safety constraints, fail-safe mechanisms, verification, and hardware-in-the-loop validation.

No license is currently included. Add a suitable `LICENSE` before public distribution; without one, copyright remains reserved by the author.
