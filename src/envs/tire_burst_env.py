import argparse
import numpy as np
import torch
import random
import time
from torch.utils.tensorboard import SummaryWriter
from carsim_env import CarSimEnv
from lpvmpc import LPVMPC
from PPO_agent import PPO
from parameters import vehicle_parameters
from reward_model import compute_reward
from utils import set_seed, Normalize

class TireBurstEnv(CarSimEnv): 

    def __init__(self, sim_path: str):
        super().__init__(sim_path)

    def reset(self):
        self.burst_tire = -1#random.randint(1, 4)
        return super().reset()
    
    def step(self, action, inner_steps=1):
        self.import_vars[0] = action.item()  # 从（1,）的numpy数组中提取标量值
        if self.t_current >= 2:
            self.import_vars[1] = self.burst_tire  # delta_z 和 爆胎车轮
        else:
            self.import_vars[1] = 0  # 前5秒不施加控制输入，但仍指定哪个轮子爆胎
        export_vars, done, info = super().step(self.import_vars, inner_steps) # export_vars 是self.export_np（list类型）的快照
        reward = compute_reward(export_vars)

        return export_vars, reward, done, info
    
    def close(self):
        super().close()

def initialize():
    p = argparse.ArgumentParser(description="PPO training for tire burst control")
    p.add_argument("--mode", type=str, default="train", help="train or test")
    p.add_argument("--actor_learning_rate", type=float, default=1e-4, help="Learning rate for PPO agent")
    p.add_argument("--critic_learning_rate", type=float, default=1e-3, help="Learning rate for PPO agent")
    p.add_argument("--gamma", type=float, default=1, help="Discount factor for PPO agent")
    p.add_argument("--lmbda", type=float, default=0.95, help="GAE lambda for PPO agent")
    p.add_argument("--eps_clip", type=float, default=0.2, help="Clipping epsilon for PPO agent")
    p.add_argument("--K_epochs", type=int, default=10, help="Number of epochs for PPO update")
    p.add_argument("--max_episodes", type=int, default=1000, help="Maximum number of training episodes")
    return p.parse_args()

def test(env, mpc_controller, agent, inner_steps_per_control=1):
    agent.reload_model('ppo_checkpoint_episode.pth')
    obs = env.reset()
    state = list(obs[:3] + (obs[5],))
    state = Normalize(state)  # 归一化状态输入    
    episode_return = 0
    action = action_mpc = action_RL = 0.0
    r_ref = np.zeros((2, mpc_controller.N))
    done = False
    while not done:
        action_mpc = mpc_controller.solve(obs, r_ref, -1) # MPC用的是原始数据
        action_RL, _ = agent.select_action(state) # 去掉std，直接使用均值作为动作
        action = action_mpc + action_RL  # 将 MPC 和 RL 的动作叠加
        obs, reward, done, _ = env.step(action, inner_steps_per_control)
        episode_return += reward
    print(f"Test episode - Return: {episode_return:.3f}")

def train(env, mpc_controller, agent, num_episodes, inner_steps_per_control=1):
    writer = SummaryWriter(f"runs/ppo_tireburst_{time.strftime('%Y%m%d-%H%M%S')}")
    return_list = []

    for ep in range(num_episodes):
        obs = env.reset() # 用于仿真迭代
        state = list(obs[:3]) # 用于训练RL
        state = Normalize(state)

        discount_factor = 1
        gamma = agent.gamma
        action = np.zeros(1)  # 初始化动作, RL返回的是nparray
        episode_return = 0
        r_ref = np.zeros((2,mpc_controller.N))
        done = False
        transition_dict = {'states': [],'actions': [],'next_states': [],'rewards': [],'dones': [],'log_probs': []}

        while not done:
            if obs[5] != 0: # 如果检测到爆胎，立即停止当前 episode
                action_mpc = mpc_controller.solve(obs, r_ref, -1)
                action_RL, log_prob = agent.select_action(state)
                writer.add_scalars(f"Action/episode_{ep}",{"MPC": action_mpc, "RL": action_RL},env.t_current*1000)
                action = action_mpc + action_RL # (1,) npshape

            next_obs, reward, done, info = env.step(action, inner_steps_per_control)
            next_state = list(next_obs[:3])
            next_state = Normalize(next_state)

            if obs[5] != 0:
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action_RL)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                transition_dict['log_probs'].append(log_prob)

            obs = next_obs
            episode_return += reward * discount_factor
            discount_factor *= gamma

        return_list.append(episode_return)
        writer.add_scalar("Reward/Episode_Return", episode_return, ep)
        print(f"Episode {ep+1}/{num_episodes} - Return: {episode_return:.3f}")
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
    set_seed(259)
    env = TireBurstEnv(sim_path="./simfile.sim")
    args = initialize()
    mpc_controller = LPVMPC(N=15, dt=0.02, params=vehicle_parameters)
    agent = PPO(
        state_dim=3, action_dim=1, hidden_dim=256,
        actor_lr=args.actor_learning_rate, critic_lr=args.critic_learning_rate,
        gamma=args.gamma, lmbda=args.lmbda, epochs=args.K_epochs, eps=args.eps_clip,
        device=torch.device("cuda") if torch.cuda.is_available() else "cpu")

    inner_steps_per_control = 20

    if args.mode == "test":
        test(env, mpc_controller, agent, inner_steps_per_control) 

    elif args.mode == "train":
        train(env, mpc_controller, agent, args.max_episodes, inner_steps_per_control)
        agent.save_model('ppo_checkpoint_episode.pth', episode=args.max_episodes)
