import argparse
import numpy as np
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from src.envs.tire_burst_env import TireBurstEnv
from src.agents.PPO_agent import PPO
from src.models.state_model import STATE_DIM, build_state
from src.utils.utils import set_seed


ACTION_NAMES = ("steer", "L1", "R1", "L2", "R2")

def log_action(writer, action, step, prefix="Action"):
    for name, value in zip(ACTION_NAMES, action):
        writer.add_scalar(f"{prefix}/{name}", float(value), step)


def initialize():
    p = argparse.ArgumentParser(description="PPO training for tire burst control")
    p.add_argument("--mode", choices=("train", "test"), default="train", help="train or test")
    p.add_argument("--actor_learning_rate", type=float, default=1e-4, help="Learning rate for PPO agent")
    p.add_argument("--critic_learning_rate", type=float, default=1e-3, help="Learning rate for PPO agent")
    p.add_argument("--gamma", type=float, default=0.995, help="Discount factor for PPO agent")
    p.add_argument("--lmbda", type=float, default=0.95, help="GAE lambda for PPO agent")
    p.add_argument("--eps_clip", type=float, default=0.2, help="Clipping epsilon for PPO agent")
    p.add_argument("--K_epochs", type=int, default=4, help="Number of epochs for PPO update")
    p.add_argument("--entropy_coef", type=float, default=1e-3, help="Entropy bonus coefficient")
    p.add_argument("--max_episodes", type=int, default=1000, help="Maximum number of training episodes")
    p.add_argument("--checkpoint_interval", type=int, default=50, help="Episodes between checkpoints")
    p.add_argument("--checkpoint", default="ppo_checkpoint_mdp11.pth", help="Checkpoint path")
    return p.parse_args()


def test(env, agent, inner_steps_per_control=1):
    writer = SummaryWriter(f"runs/ppo_tireburst_{time.strftime('%Y%m%d-%H%M%S')}")
    agent.actor.eval()
    agent.critic.eval()

    obs = env.reset(0)
    previous_action = np.zeros(5, dtype=np.float32)
    state = build_state(obs, previous_action)
    episode_return = 0.0
    done = False
    control_step = 0

    while not done:
        if obs[5] != 0:
            action, _ = agent.select_action(state, deterministic=True)
            log_action(writer, action, control_step, prefix="Action_test")
        else:
            action = np.zeros(5, dtype=np.float32)

        next_obs, reward, done, info = env.step(action, inner_steps_per_control)
        obs = next_obs
        previous_action = action
        state = build_state(obs, previous_action)
        episode_return += reward
        control_step += 1

    print(f"Test episode - Return: {episode_return:.3f}")
    writer.close()


def train(
    env,
    agent,
    num_episodes,
    inner_steps_per_control=1,
    checkpoint_interval=50,
    checkpoint_path="ppo_checkpoint_mdp11.pth",
):
    writer = SummaryWriter(f"runs/ppo_tireburst_{time.strftime('%Y%m%d-%H%M%S')}")
    return_list = []
    global_control_step = 0
    agent.actor.train()
    agent.critic.train()

    for ep in range(num_episodes):
        obs = env.reset(ep)
        previous_action = np.zeros(5, dtype=np.float32)
        state = build_state(obs, previous_action)
        episode_return = 0.0
        done = False
        reward_sums = {
            "track": 0.0,
            "stability": 0.0,
            "control": 0.0,
            "smoothness": 0.0,
        }
        transition_dict = {
            "states": [],
            "actions": [],
            "next_states": [],
            "rewards": [],
            "dones": [],
            "log_probs": [],
        }

        while not done:
            burst_active = obs[5] != 0
            if burst_active:
                decision_state = state.copy()
                action, log_prob = agent.select_action(state)
                log_action(writer, action, global_control_step)
            else:
                action = np.zeros(5, dtype=np.float32)
                log_prob = None

            next_obs, reward, done, info = env.step(action, inner_steps_per_control)
            next_state = build_state(next_obs, action)
            episode_return += reward
            for name in reward_sums:
                reward_sums[name] += info[f"reward_{name}"]

            if burst_active:
                transition_dict["states"].append(decision_state)
                transition_dict["actions"].append(action.copy())
                transition_dict["next_states"].append(next_state)
                transition_dict["rewards"].append(reward)
                transition_dict["dones"].append(info["terminated"])
                transition_dict["log_probs"].append(log_prob)

            obs = next_obs
            state = next_state
            previous_action = action
            global_control_step += 1

        return_list.append(episode_return)
        writer.add_scalar("Reward/Episode_Return", episode_return, ep)
        for name, value in reward_sums.items():
            writer.add_scalar(f"RewardComponent/{name}", value, ep)
        writer.add_scalar("Curriculum/Stage", env.curriculum_stage, ep)
        writer.add_scalar("Curriculum/Speed", env.v, ep)
        writer.add_scalar("Curriculum/Mu", env.mu, ep)
        print(f"Episode {ep+1}/{num_episodes} - Return: {episode_return:.3f}")
        if len(return_list) >= 10:
            writer.add_scalar("Reward/Moving_Avg_10", np.mean(return_list[-10:]), ep)

        metrics = agent.update(transition_dict)
        if metrics is not None:
            writer.add_scalar("Loss/Actor", metrics["actor_loss"], ep)
            writer.add_scalar("Loss/Critic", metrics["critic_loss"], ep)
            writer.add_scalar("Policy/Entropy", metrics["entropy"], ep)
            writer.add_scalar("Policy/ClipFrac", metrics["clip_frac"], ep)
            writer.add_scalar("Policy/NormalizedStd", metrics["normalized_std"], ep)

        writer.flush()
        if checkpoint_interval > 0 and (ep + 1) % checkpoint_interval == 0:
            agent.save_model(checkpoint_path, episode=ep)

    writer.close()


if __name__ == "__main__":
    set_seed(7)
    env = TireBurstEnv(sim_path="./simfile.sim")
    args = initialize()
    agent = PPO(
        state_dim=STATE_DIM,
        env_dim=0,
        action_dim=5,
        hidden_dim=256,
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

    elif args.mode == "train":
        train(
            env,
            agent,
            args.max_episodes,
            inner_steps_per_control,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_path=args.checkpoint,
        )
        agent.save_model(args.checkpoint, episode=args.max_episodes - 1)
