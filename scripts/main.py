import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.agents.PPO_agent import PPO
from src.agents.history_encoder import AdaptationDataset, HistoryBuffer
from src.envs.tire_burst_env import TireBurstEnv
from src.models.state_model import STATE_DIM, build_state


ACTION_NAMES = ("steer", "L1", "R1", "L2", "R2")


def log_action(writer, action, step, prefix="Action"):
    for name, value in zip(ACTION_NAMES, action):
        writer.add_scalar(f"{prefix}/{name}", float(value), step)


def make_transition_dict():
    return {
        "states": [],
        "env_infos": [],
        "actions": [],
        "raw_actions": [],
        "next_states": [],
        "rewards": [],
        "terminateds": [],
        "dones": [],
        "log_probs": [],
    }


def initialize():
    parser = argparse.ArgumentParser(
        description="PPO training for tire burst control"
    )
    parser.add_argument(
        "--mode",
        choices=(
            "train",
            "resume",
            "test",
            "collect_history_data",
            "test_history_encoder",
        ),
        default="train",
    )
    parser.add_argument("--actor_learning_rate", type=float, default=3e-4)
    parser.add_argument("--critic_learning_rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--K_epochs", type=int, default=3)
    parser.add_argument("--entropy_coef", type=float, default=1e-4)
    parser.add_argument("--max_episodes", type=int, default=4000)
    parser.add_argument("--adaptation_episodes", type=int, default=100)
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset")
    parser.add_argument("--history_checkpoint")
    return parser.parse_args()


def test(env, agent, inner_steps_per_control=1):
    agent.actor.eval()
    agent.privileged_encoder.eval()
    obs, env_info = env.reset(0, curriculum_active=False)
    state = build_state(
        obs, np.zeros(5, dtype=np.float32), target_vx=obs[0]
    )
    episode_return = 0.0
    done = False

    while not done:
        if env.burst_active:
            action, _, _ = agent.select_action_oracle(
                state, env_info, deterministic=True
            )
        else:
            action = np.zeros(5, dtype=np.float32)
        next_obs, env_info, reward, done, _ = env.step(
            action, inner_steps_per_control
        )
        target_vx = env.target_vx if env.burst_active else next_obs[0]
        state = build_state(next_obs, action, target_vx=target_vx)
        episode_return += reward

    print(f"Oracle test return: {episode_return:.3f}")


def train(env, agent, num_episodes, inner_steps_per_control=1,
    checkpoint_interval=100,
    checkpoint_path="ppo_checkpoint_latent.pth",
    start_episode=0,
):
    writer = SummaryWriter(f"runs/ppo_v3/ppo_tireburst_{time.strftime('%Y%m%d-%H%M%S')}")
    global_control_step = 0
    agent.actor.train()
    agent.privileged_encoder.train()
    agent.critic.train()
    transition_dict = make_transition_dict()
    end_episode = start_episode + num_episodes

    for episode in range(start_episode, end_episode):
        obs, env_info = env.reset(episode)
        state = build_state(
            obs, np.zeros(5, dtype=np.float32), target_vx=obs[0]
        )
        episode_return = 0.0
        done = False
        metrics = None
        reward_sums = {
            "speed": 0.0,
            "track": 0.0,
            "stability": 0.0,
            "control": 0.0,
            "smoothness": 0.0,
            "terminal": 0.0,
        }

        while not done:
            policy_controlled = env.burst_active
            if policy_controlled:
                action, log_prob, raw_action = agent.select_action_oracle(state, env_info)
                log_action(writer, action, global_control_step)
            else:
                action = np.zeros(5, dtype=np.float32)
                log_prob = None

            next_obs, env_info, reward, done, info = env.step(
                action, inner_steps_per_control
            )
            target_vx = env.target_vx if env.burst_active else next_obs[0]
            next_state = build_state(next_obs, action, target_vx=target_vx)
            episode_return += reward
            for name in reward_sums:
                reward_sums[name] += info[f"reward_{name}"]

            if policy_controlled:
                transition_dict["states"].append(state)
                transition_dict["env_infos"].append(env_info)
                transition_dict["actions"].append(action.copy())
                transition_dict["raw_actions"].append(raw_action.copy())
                transition_dict["next_states"].append(next_state)
                transition_dict["rewards"].append(reward)
                transition_dict["terminateds"].append(info["terminated"])
                transition_dict["dones"].append(done)
                transition_dict["log_probs"].append(log_prob)

            state = next_state
            global_control_step += 1

        writer.add_scalar("Reward/Episode_Return", episode_return, episode)
        for name, value in reward_sums.items():
            writer.add_scalar(f"RewardComponent/{name}", value, episode)

        print(
            f"Episode {episode + 1}/{end_episode} - "
            f"Return: {episode_return:.3f}"
        )

        if (episode - start_episode + 1) % 4 == 0:
            metrics = agent.update(transition_dict)
            transition_dict = make_transition_dict()
        if metrics is not None:
            writer.add_scalar("Loss/Actor", metrics["actor_loss"], episode)
            writer.add_scalar("Loss/Critic", metrics["critic_loss"], episode)
            writer.add_scalar("Policy/Entropy", metrics["entropy"], episode)
            writer.add_scalar("Policy/ClipFrac", metrics["clip_frac"], episode)
            writer.add_scalar(
                "Policy/NormalizedStd", metrics["normalized_std"], episode
            )
        writer.flush()

        if checkpoint_interval > 0 and (episode + 1) % checkpoint_interval == 0:
            agent.save_model(checkpoint_path, episode=episode)

    writer.close()
    return end_episode - 1


def collect_history_data(env, agent, dataset_path, inner_steps_per_control=1, num_episodes=500,):
    agent.actor.eval()
    agent.privileged_encoder.eval()
    agent.history_encoder.eval()
    histories = []
    labels = []

    for episode in range(num_episodes):
        obs, env_info = env.reset(episode, curriculum_active=False)
        state = build_state(
            obs, np.zeros(5, dtype=np.float32), target_vx=obs[0]
        )
        history_buffer = HistoryBuffer(maxlen=25)
        history_buffer.reset(state)
        done = False

        while not done:
            policy_controlled = env.burst_active
            history = history_buffer.as_array()
            if policy_controlled:
                action, _, _ = agent.select_action_oracle(state, env_info)
                with torch.no_grad():
                    target_z = (agent.encode_env_info(env_info).squeeze(0).cpu().numpy())
            else:
                action = np.zeros(5, dtype=np.float32)

            next_obs, env_info, _, done, _ = env.step(
                action, inner_steps_per_control
            )
            target_vx = env.target_vx if env.burst_active else next_obs[0]
            next_state = build_state(next_obs, action, target_vx=target_vx)

            if policy_controlled:
                histories.append(history.copy())
                labels.append(target_z.copy())

            history_buffer.append(next_state)
            state = next_state

        print(f"Collected episode {episode + 1}/{num_episodes}")

    dataset = AdaptationDataset(histories, labels)
    dataset.save(dataset_path)
    print(f"Saved {len(dataset)} samples to {dataset_path}")


def test_history_encoder(env, agent, inner_steps_per_control=1, num_episodes=1):
    agent.actor.eval()
    agent.privileged_encoder.eval()
    agent.history_encoder.eval()

    for episode in range(num_episodes):
        obs, env_info = env.reset(episode, curriculum_active=False)
        state = build_state(
            obs, np.zeros(5, dtype=np.float32), target_vx=obs[0]
        )
        history_buffer = HistoryBuffer(maxlen=25)
        history_buffer.reset(state)
        done = False
        episode_return = 0.0

        while not done:
            if env.burst_active:
                action, _, _ = agent.select_action_adaptive(
                    state, history_buffer.as_array(), deterministic=True
                )
            else:
                action = np.zeros(5, dtype=np.float32)

            next_obs, env_info, reward, done, _ = env.step(
                action, inner_steps_per_control
            )
            target_vx = env.target_vx if env.burst_active else next_obs[0]
            next_state = build_state(next_obs, action, target_vx=target_vx)
            history_buffer.append(next_state)
            state = next_state
            episode_return += reward

        print(
            f"Adaptive test episode {episode + 1}/{num_episodes}: "
            f"return={episode_return:.3f}"
        )


def load_history_encoder(agent, checkpoint):
    try:
        state_dict = torch.load(
            checkpoint, map_location=agent.device, weights_only=True
        )
    except TypeError:
        state_dict = torch.load(checkpoint, map_location=agent.device)
    agent.history_encoder.load_state_dict(state_dict)


if __name__ == "__main__":
    args = initialize()
    env = TireBurstEnv(sim_path="./simfile.sim")
    agent = PPO(
        state_dim=STATE_DIM,
        env_dim=6,
        action_dim=5,
        hidden_dim=256,
        latent_dim=8,
        actor_lr=args.actor_learning_rate,
        critic_lr=args.critic_learning_rate,
        gamma=args.gamma,
        lmbda=args.lmbda,
        epochs=args.K_epochs,
        eps=args.eps_clip,
        device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
        entropy_coef=args.entropy_coef,
    )
    inner_steps_per_control = 20

    if args.mode == "test":
        agent.reload_model(args.checkpoint)
        test(env, agent, inner_steps_per_control)

    elif args.mode in ("train", "resume"):
        start_episode = 0
        if args.mode == "resume":
            start_episode = agent.reload_model(args.checkpoint)
            print(
                f"Resuming from {args.checkpoint}: "
                f"next curriculum episode is {start_episode}."
            )
        last_episode = train(
            env,
            agent,
            args.max_episodes,
            inner_steps_per_control,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_path=args.checkpoint,
            start_episode=start_episode,
        )
        agent.save_model(args.checkpoint, episode=last_episode)

    elif args.mode == "collect_history_data":
        agent.reload_model(args.checkpoint)
        if args.history_checkpoint and Path(args.history_checkpoint).is_file():
            load_history_encoder(agent, args.history_checkpoint)
        else:
            print("History checkpoint not found; collecting with a random CNN.")
        collect_history_data(
            env,
            agent,
            args.dataset,
            inner_steps_per_control,
            num_episodes=args.adaptation_episodes,
        )

    elif args.mode == "test_history_encoder":
        agent.reload_model(args.checkpoint)
        load_history_encoder(agent, args.history_checkpoint)
        test_history_encoder(
            env,
            agent,
            inner_steps_per_control,
            num_episodes=args.adaptation_episodes,
        )

    env.close()
