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

def make_transition_dict():
    return {
        "states": [],
        "env_infos": [],
        "actions": [],
        "next_states": [],
        "rewards": [],
        "terminateds": [],
        "dones": [],
        "log_probs": [],
    }

def initialize():
    p = argparse.ArgumentParser(description="PPO training for tire burst control")
    p.add_argument(
        "--mode",
        choices=("train", "resume", "test"),
        default="train",
        help="start fresh training, resume from a checkpoint, or test",
    )
    p.add_argument("--actor_learning_rate", type=float, default=1e-4, help="Learning rate for PPO agent")
    p.add_argument("--critic_learning_rate", type=float, default=1e-3, help="Learning rate for PPO agent")
    p.add_argument("--gamma", type=float, default=0.995, help="Discount factor for PPO agent")
    p.add_argument("--lmbda", type=float, default=0.95, help="GAE lambda for PPO agent")
    p.add_argument("--eps_clip", type=float, default=0.2, help="Clipping epsilon for PPO agent")
    p.add_argument("--K_epochs", type=int, default=4, help="Number of epochs for PPO update")
    p.add_argument("--entropy_coef", type=float, default=1e-3, help="Entropy bonus coefficient")
    p.add_argument(
        "--max_episodes",
        type=int,
        default=1000,
        help="Number of episodes to run in this invocation",
    )
    p.add_argument("--checkpoint_interval", type=int, default=100, help="Episodes between checkpoints")
    p.add_argument("--checkpoint", default="ppo_checkpoint_latent.pth", help="Checkpoint path")
    return p.parse_args()


def test(env, agent, inner_steps_per_control=1):
    writer = SummaryWriter(f"runs/ppo_tireburst_{time.strftime('%Y%m%d-%H%M%S')}")
    agent.actor.eval()
    agent.privileged_encoder.eval()
    agent.critic.eval()

    obs, env_info = env.reset(0)
    previous_action = np.zeros(5, dtype=np.float32)
    state = build_state(obs, previous_action, target_vx=obs[0])
    episode_return = 0.0
    done = False
    control_step = 0

    while not done:
        # if obs[5] != 4:
        #     action, _ = agent.select_action(state, deterministic=True)
        #     log_action(writer, action, control_step, prefix="Action_test")
        # else:
        action = np.zeros(5, dtype=np.float32)

        next_obs, env_info, reward, done, info = env.step(action, inner_steps_per_control)
        obs = next_obs
        previous_action = action
        target_vx = env.target_vx if env.burst_active else obs[0]
        state = build_state(obs, previous_action, target_vx=target_vx)
        episode_return += reward
        control_step += 1

    print(f"Test episode - Return: {episode_return:.3f}")
    writer.close()


def train(
    env,
    agent,
    num_episodes,
    inner_steps_per_control=1,
    checkpoint_interval=100,
    checkpoint_path="ppo_checkpoint_latent.pth",
    start_episode=0,
):
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")
    if start_episode < 0:
        raise ValueError("start_episode must be non-negative.")

    writer = SummaryWriter(f"runs/ppo_tireburst_{time.strftime('%Y%m%d-%H%M%S')}")
    return_list = []
    global_control_step = 0
    agent.actor.train()
    agent.privileged_encoder.train()
    agent.critic.train()

    transition_dict = make_transition_dict()
    end_episode = start_episode + num_episodes

    for ep in range(start_episode, end_episode):
        obs, env_info = env.reset(ep)
        previous_action = np.zeros(5, dtype=np.float32)
        state = build_state(obs, previous_action, target_vx=obs[0])
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
                action, log_prob = agent.select_action(state, env_info)
                log_action(writer, action, global_control_step)
            else:
                action = np.zeros(5, dtype=np.float32)
                log_prob = None

            next_obs, env_info, reward, done, info = env.step(action, inner_steps_per_control)
            target_vx = env.target_vx if env.burst_active else next_obs[0]
            next_state = build_state(next_obs, action, target_vx=target_vx)
            episode_return += reward
            for name in reward_sums:
                reward_sums[name] += info[f"reward_{name}"]

            if policy_controlled:
                transition_dict["states"].append(state)
                transition_dict["env_infos"].append(env_info)
                transition_dict["actions"].append(action.copy())
                transition_dict["next_states"].append(next_state)
                transition_dict["rewards"].append(reward)
                transition_dict["terminateds"].append(info["terminated"])
                transition_dict["dones"].append(done)
                transition_dict["log_probs"].append(log_prob)

            obs = next_obs
            state = next_state
            previous_action = action
            global_control_step += 1

        return_list.append(episode_return)
        writer.add_scalar("Reward/Episode_Return", episode_return, ep)
        for name, value in reward_sums.items():
            writer.add_scalar(f"RewardComponent/{name}", value, ep)

        print(f"Episode {ep+1}/{end_episode} - Return: {episode_return:.3f}")

        episodes_collected = ep - start_episode + 1
        if episodes_collected % 4 == 0:
            metrics = agent.update(transition_dict)
            transition_dict = make_transition_dict()
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
    return end_episode - 1


if __name__ == "__main__":
    # set_seed(7)
    env = TireBurstEnv(sim_path="./simfile.sim")
    args = initialize()
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
        # agent.reload_model(args.checkpoint)
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
