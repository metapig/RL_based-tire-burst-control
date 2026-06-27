import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.PPO_agent import PPO
from src.agents.mlp_env_encoder import MLPEnvEncoder
from src.envs.tire_burst_env import TireBurstEnv
from src.models.parameters import vehicle_parameters
from src.mpc.lpvmpc import LPVMPC
from src.utils.history_buffer import HistoryBuffer
from src.utils.utils import build_policy_state, set_seed


def initialize():
    parser = argparse.ArgumentParser(description="PPO training for tire burst control")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="train or test")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-4, help="Actor learning rate")
    parser.add_argument("--critic_learning_rate", type=float, default=1e-3, help="Critic learning rate")
    parser.add_argument("--gamma", type=float, default=1, help="Discount factor")
    parser.add_argument("--lmbda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument("--K_epochs", type=int, default=4, help="Number of PPO epochs per update")
    parser.add_argument("--max_episodes", type=int, default=1000, help="Maximum number of training episodes")
    parser.add_argument(
        "--mu_source",
        type=str,
        default="true",
        choices=["true", "encoder"],
        help="Use true env.mu or encoder-estimated mu_hat in PPO input",
    )
    parser.add_argument(
        "--encoder_checkpoint",
        type=str,
        default="",
        help="Path to MLP encoder checkpoint used when --mu_source encoder",
    )
    parser.add_argument("--encoder_history_len", type=int, default=25, help="Online encoder history length")
    return parser.parse_args()


def build_state(obs, env, mu_source, history_buffer=None, encoder=None, device=None, writer=None, step=None):
    if mu_source == "true":
        return build_policy_state(obs, env.mu), float(env.mu)

    if history_buffer is None or encoder is None:
        raise ValueError("history_buffer and encoder are required when mu_source='encoder'.")

    mu_hat = encoder.predict_mu(history_buffer.as_array(), device=device)
    if writer is not None and step is not None:
        writer.add_scalar("EnvEncoder/mu_hat", mu_hat, step)
        writer.add_scalar("EnvEncoder/mu_true", float(env.mu), step)
        writer.add_scalar("EnvEncoder/mu_abs_error", abs(mu_hat - float(env.mu)), step)
    return build_policy_state(obs, mu_hat), mu_hat


def make_history_buffer(mu_source, history_len):
    return HistoryBuffer(history_len=history_len) if mu_source == "encoder" else None


def test(env, mpc_controller, agent, inner_steps_per_control=1, mu_source="true", encoder=None, device=None,
         encoder_history_len=25):
    writer = SummaryWriter(f"runs/ppo_tireburst_{time.strftime('%Y%m%d-%H%M%S')}")
    obs = env.reset()

    episode_return = 0.0
    k_counter = 10
    action_mpc = 0.0
    alpha = np.zeros(3, dtype=np.float32)
    r_ref = np.zeros((2, mpc_controller.N))
    history_buffer = make_history_buffer(mu_source, encoder_history_len)
    if history_buffer is not None:
        history_buffer.append(obs, action_mpc, alpha)
    state, _ = build_state(obs, env, mu_source, history_buffer, encoder, device, writer, 0)

    done = False
    while not done:
        if obs[5] != 0:
            if k_counter == 10:
                alpha, _ = agent.select_action(state)
                alpha = np.array(alpha).reshape(-1)

            alpha_q = alpha[:2]
            alpha_r = alpha[2:].reshape(-1)
            action_mpc = mpc_controller.solve(obs, r_ref, -1, alpha_q, alpha_r)
            writer.add_scalar("Action_test", action_mpc, env.t_current * 1000)

        next_obs, reward, done, _ = env.step(action_mpc, alpha, inner_steps_per_control)
        obs = next_obs
        if history_buffer is not None:
            history_buffer.append(obs, action_mpc, alpha)
        state, _ = build_state(obs, env, mu_source, history_buffer, encoder, device, writer, env.t_current)
        episode_return += reward

        k_counter += 1
        if k_counter > 10:
            k_counter = 0

    writer.close()
    print(f"Test episode - Return: {episode_return:.3f}")


def train(env, mpc_controller, agent, num_episodes, inner_steps_per_control=1, mu_source="true", encoder=None,
          device=None, encoder_history_len=25):
    writer = SummaryWriter(f"runs/ppo_tireburst_{time.strftime('%Y%m%d-%H%M%S')}")
    return_list = []

    for ep in range(num_episodes):
        obs = env.reset()

        k_counter = 10
        action_mpc = 0.0
        alpha = np.zeros(3, dtype=np.float32)
        episode_return = 0.0
        k_step_reward = 0.0
        r_ref = np.zeros((2, mpc_controller.N))
        done = False
        transition_dict = {"states": [], "actions": [], "next_states": [], "rewards": [], "dones": [], "log_probs": []}
        history_buffer = make_history_buffer(mu_source, encoder_history_len)
        if history_buffer is not None:
            history_buffer.append(obs, action_mpc, alpha)
        state, _ = build_state(obs, env, mu_source, history_buffer, encoder, device, writer, ep)

        while not done:
            if obs[5] != 0:
                if k_counter == 10:
                    decision_state = state.copy()
                    alpha, log_prob = agent.select_action(state)
                    alpha = np.array(alpha).reshape(-1)
                    k_step_reward = 0.0

                alpha_q = alpha[:2]
                alpha_r = alpha[2:].reshape(-1)
                action_mpc = mpc_controller.solve(obs, r_ref, -1, alpha_q, alpha_r)
                writer.add_scalar(f"Action/episode_{ep}", action_mpc, env.t_current * 1000)

            next_obs, reward, done, _ = env.step(action_mpc, alpha, inner_steps_per_control)

            if history_buffer is not None:
                history_buffer.append(next_obs, action_mpc, alpha)
            log_step = ep * 100000 + int(env.t_current * 1000)
            next_state, _ = build_state(next_obs, env, mu_source, history_buffer, encoder, device, writer, log_step)
            k_step_reward += reward
            episode_return += reward

            if obs[5] != 0 and k_counter == 10:
                transition_dict["states"].append(decision_state)
                transition_dict["actions"].append(alpha)
                transition_dict["next_states"].append(next_state)
                transition_dict["rewards"].append(k_step_reward)
                transition_dict["dones"].append(done)
                transition_dict["log_probs"].append(log_prob)

            obs = next_obs
            state = next_state

            k_counter += 1
            if k_counter > 10:
                k_counter = 0

        return_list.append(episode_return)
        writer.add_scalar("Reward/Episode_Return", episode_return, ep)
        print(f"Episode {ep + 1}/{num_episodes} - Return: {episode_return:.3f}")
        if len(return_list) >= 10:
            writer.add_scalar("Reward/Moving_Avg_10", np.mean(return_list[-10:]), ep)

        metrics = agent.update(transition_dict)
        if metrics is not None:
            writer.add_scalar("Loss/Actor", metrics["actor_loss"], ep)
            writer.add_scalar("Loss/Critic", metrics["critic_loss"], ep)
            writer.add_scalar("Policy/Entropy", metrics["entropy"], ep)
            writer.add_scalar("Policy/ClipFrac", metrics["clip_frac"], ep)

        writer.flush()

    writer.close()


if __name__ == "__main__":
    set_seed(7)
    args = initialize()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    env = TireBurstEnv(sim_path="./simfile.sim")
    mpc_controller = LPVMPC(N=15, dt=0.02, params=vehicle_parameters)
    agent = PPO(
        state_dim=3,
        env_dim=1,
        action_dim=3,
        hidden_dim=256,
        actor_lr=args.actor_learning_rate,
        critic_lr=args.critic_learning_rate,
        gamma=args.gamma,
        lmbda=args.lmbda,
        epochs=args.K_epochs,
        eps=args.eps_clip,
        device=device,
    )

    encoder = None
    if args.mu_source == "encoder":
        if not args.encoder_checkpoint:
            raise ValueError("--encoder_checkpoint is required when --mu_source encoder.")
        encoder = MLPEnvEncoder.from_checkpoint(args.encoder_checkpoint, map_location=device).to(device)
        encoder.eval()

    inner_steps_per_control = 20
    if args.mode == "test":
        agent.reload_model("ppo_checkpoint_episode2.pth")
        test(
            env,
            mpc_controller,
            agent,
            inner_steps_per_control,
            args.mu_source,
            encoder,
            device,
            args.encoder_history_len,
        )
    elif args.mode == "train":
        train(
            env,
            mpc_controller,
            agent,
            args.max_episodes,
            inner_steps_per_control,
            args.mu_source,
            encoder,
            device,
            args.encoder_history_len,
        )
        agent.save_model("ppo_checkpoint_episode2.pth", episode=args.max_episodes)
