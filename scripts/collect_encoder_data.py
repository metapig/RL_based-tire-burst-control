import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.PPO_agent import PPO
from src.envs.tire_burst_env import TireBurstEnv
from src.models.parameters import vehicle_parameters
from src.mpc.lpvmpc import LPVMPC
from src.utils.history_buffer import HISTORY_FEATURE_NAMES, HistoryBuffer
from src.utils.utils import build_policy_state, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Collect supervised data for the RMA-style mu encoder.")
    parser.add_argument("--sim_path", type=str, default="./simfile.sim")
    parser.add_argument("--output", type=str, default="runs/encoder_data/encoder_dataset.npz")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps_per_episode", type=int, default=20000)
    parser.add_argument("--history_len", type=int, default=25)
    parser.add_argument("--inner_steps_per_control", type=int, default=20)
    parser.add_argument("--ppo_checkpoint", type=str, default="ppo_checkpoint_episode2.pth")
    parser.add_argument("--use_checkpoint", action="store_true", help="Load PPO checkpoint before collection.")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    env = TireBurstEnv(sim_path=args.sim_path)
    mpc_controller = LPVMPC(N=15, dt=0.02, params=vehicle_parameters)
    agent = PPO(
        state_dim=3,
        env_dim=1,
        action_dim=3,
        hidden_dim=256,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=1,
        lmbda=0.95,
        epochs=4,
        eps=0.2,
        device=device,
    )
    if args.use_checkpoint:
        agent.reload_model(args.ppo_checkpoint)

    histories = []
    labels = []
    episode_ids = []
    times = []

    for ep in range(args.episodes):
        obs = env.reset()
        history_buffer = HistoryBuffer(history_len=args.history_len)
        action_mpc = 0.0
        alpha = np.zeros(3, dtype=np.float32)
        history_buffer.append(obs, action_mpc)
        state = build_policy_state(obs, env.mu)
        r_ref = np.zeros((2, mpc_controller.N))
        k_counter = 10
        done = False
        steps = 0

        while not done and steps < args.max_steps_per_episode:
            if obs[5] != 0:
                if k_counter == 10:
                    alpha, _ = agent.select_action(state)
                    alpha = np.array(alpha).reshape(-1)
                alpha_q = alpha[:2]
                alpha_r = alpha[2:].reshape(-1)
                action_mpc = mpc_controller.solve(obs, r_ref, -1, alpha_q, alpha_r)

            next_obs, _, done, _ = env.step(action_mpc, alpha, args.inner_steps_per_control)
            history_buffer.append(next_obs, action_mpc)
            histories.append(history_buffer.as_array()) # 一个episode中一组训练数据的一个完整输入
            labels.append(float(env.mu))
            episode_ids.append(ep)
            times.append(float(env.t_current))

            obs = next_obs
            state = build_policy_state(obs, env.mu)
            steps += 1
            k_counter += 1
            if k_counter > 10:
                k_counter = 0

        print(f"Episode {ep + 1}/{args.episodes}: steps={steps}, mu={float(env.mu):.3f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        histories=np.asarray(histories, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int32),
        times=np.asarray(times, dtype=np.float32),
        feature_names=np.asarray(HISTORY_FEATURE_NAMES),
        history_len=np.asarray(args.history_len, dtype=np.int32),
    )
    print(f"Saved {len(labels)} samples to {output_path}")


if __name__ == "__main__":
    main()
