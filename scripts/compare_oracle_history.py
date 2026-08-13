import argparse

import numpy as np
import torch

from src.agents.PPO_agent import PPO
from src.agents.history_encoder import HistoryBuffer
from src.envs.tire_burst_env import TireBurstEnv
from src.models.state_model import STATE_DIM, build_state


def run_episode(env, agent, episode, initial_params, use_history):
    obs, env_info = env.reset(
        episode,
        curriculum_active=False,
        initial_params=initial_params,
    )
    actual_params = {
        "v": env.v,
        "mu": env.mu,
        "m": env.m,
        "Iz": env.Iz,
        "burst_severity": env.burst_severity,
        "burst_tire": env.burst_tire,
    }
    state = build_state(
        obs, np.zeros(5, dtype=np.float32), target_vx=obs[0]
    )
    history_buffer = HistoryBuffer(maxlen=25)
    history_buffer.reset(state)

    with torch.no_grad():
        oracle_z = agent.encode_env_info(env_info).squeeze(0).cpu().numpy()

    predicted_zs = []
    episode_return = 0.0
    done = False

    while not done:
        if env.burst_active:
            if use_history:
                history = history_buffer.as_array()
                with torch.no_grad():
                    predicted_z = (
                        agent.predict_latent(history).squeeze(0).cpu().numpy()
                    )
                predicted_zs.append(predicted_z)
                action, _, _ = agent.select_action_adaptive(
                    state, history, deterministic=True
                )
            else:
                action, _, _ = agent.select_action_oracle(
                    state, env_info, deterministic=True
                )
        else:
            action = np.zeros(5, dtype=np.float32)

        next_obs, env_info, reward, done, _ = env.step(action, 20)
        target_vx = env.target_vx if env.burst_active else next_obs[0]
        state = build_state(next_obs, action, target_vx=target_vx)
        history_buffer.append(state)
        episode_return += reward

    return episode_return, oracle_z, predicted_zs, actual_params


def load_history_encoder(agent, checkpoint):
    try:
        state_dict = torch.load(
            checkpoint, map_location=agent.device, weights_only=True
        )
    except TypeError:
        state_dict = torch.load(checkpoint, map_location=agent.device)
    agent.history_encoder.load_state_dict(state_dict)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="ppo_checkpoint_latent.pth")
    parser.add_argument("--history_checkpoint", default="history_encoder.pth")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--v", type=float, default=140.0)
    parser.add_argument("--mu", type=float, default=0.4)
    parser.add_argument("--m", type=float, default=1500.0)
    parser.add_argument("--burst_severity", type=float, default=0.75)
    parser.add_argument("--burst_tire", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    env = TireBurstEnv(sim_path="./simfile.sim")
    agent = PPO(
        state_dim=STATE_DIM,
        env_dim=6,
        action_dim=5,
        hidden_dim=256,
        latent_dim=8,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.995,
        lmbda=0.95,
        epochs=4,
        eps=0.2,
        device=device,
    )
    agent.reload_model(args.checkpoint)
    load_history_encoder(agent, args.history_checkpoint)
    agent.actor.eval()
    agent.privileged_encoder.eval()
    agent.history_encoder.eval()

    fixed_params = None
    if args.fixed:
        fixed_params = {
            "v": args.v,
            "mu": args.mu,
            "m": args.m,
            "Iz": args.m * 1.21,
            "burst_severity": args.burst_severity,
            "burst_tire": args.burst_tire,
        }

    try:
        for episode in range(args.episodes):
            oracle_return, oracle_z, _, actual_params = run_episode(
                env, agent, episode, fixed_params, use_history=False
            )
            history_return, _, predicted_zs, _ = run_episode(
                env, agent, episode, actual_params, use_history=True
            )
            predicted_zs = np.asarray(predicted_zs)
            z_rmse = np.sqrt(np.mean((predicted_zs - oracle_z) ** 2))

            print(f"Episode {episode + 1}: {actual_params}")
            print(f"Oracle z: {oracle_z}")
            print(f"History z mean: {predicted_zs.mean(axis=0)}")
            print(f"z RMSE: {z_rmse:.6f}")
            print(f"Oracle return: {oracle_return:.3f}")
            print(f"History return: {history_return:.3f}")
    finally:
        env.close()
