import argparse
import numpy as np
import torch
import random
import time
from torch.utils.tensorboard import SummaryWriter
from carsim_env import CarSimEnv
from lpvmpc import LPVMPC
from PPO_agent_copy import PPO
from parameters import vehicle_parameters
from reward_model import compute_reward
from utils import set_seed, Normalize, modify_run_all_par

class TireBurstEnv(CarSimEnv): 

    def __init__(self, sim_path: str):
        super().__init__(sim_path)

    def reset(self):
        par_path = r"D:\CarSim2023\Databases\Results\Run_60366987-856d-45b1-b359-7939e85af520\Run_all.par"
        self.burst_tire = random.choice([-2, -1, 1, 2])
        self.v = np.random.uniform(70, 100)
        self.mu = np.random.uniform(0.2, 1.0)
        self.m = np.random.uniform(1000, 2000)
        self.Iz = np.random.uniform(2000, 4000)

        # 这里做domain randomization，直接在环境层面设置 VS 的参数
        modify_run_all_par(self.v, self.mu, par_path)

        compute_reward.prev_action = None
        compute_reward.prev_alpha = None

        return super().reset()
    
    def step(self, action_mpc, alpha, inner_steps=1):

        self.import_vars[0] = action_mpc  # 从（1,）的numpy数组中提取标量值

        if self.t_current >= 2:
            self.import_vars[1] = self.burst_tire  # delta_z 和 爆胎车轮
        else:
            self.import_vars[1] = 0  # 前5秒不施加控制输入，但仍指定哪个轮子爆胎

        export_vars, done, info = super().step(self.import_vars, inner_steps) # export_vars 是self.export_np（list类型）的快照
        reward = compute_reward(export_vars, action_mpc, alpha)

        return export_vars, reward, done, info
    
    def close(self):
        super().close()

def initialize():
    p = argparse.ArgumentParser(description="PPO training for tire burst control")
    p.add_argument("--mode", type=str, default="test", help="train or test")
    p.add_argument("--actor_learning_rate", type=float, default=1e-4, help="Learning rate for PPO agent")
    p.add_argument("--critic_learning_rate", type=float, default=1e-3, help="Learning rate for PPO agent")
    p.add_argument("--gamma", type=float, default=1, help="Discount factor for PPO agent")
    p.add_argument("--lmbda", type=float, default=0.95, help="GAE lambda for PPO agent")
    p.add_argument("--eps_clip", type=float, default=0.2, help="Clipping epsilon for PPO agent")
    p.add_argument("--K_epochs", type=int, default=10, help="Number of epochs for PPO update")
    p.add_argument("--max_episodes", type=int, default=1, help="Maximum number of training episodes")
    return p.parse_args()

def test(env, mpc_controller, agent, inner_steps_per_control=1):
    writer = SummaryWriter(f"runs/ppo_tireburst_{time.strftime('%Y%m%d-%H%M%S')}")
    obs = env.reset()
    state = list(obs[:3])
    state = Normalize(state)  # 归一化状态输入    
    
    episode_return = 0
    K = 10
    action_mpc = 0
    alpha = np.zeros(3)
    r_ref = np.zeros((2, mpc_controller.N))

    done = False
    while not done:
        print(obs)
        if obs[5] != 0: 
            if K == 10: 
                alpha, log_prob = agent.select_action(state)
                alpha = np.array(alpha).reshape(-1)
                # print(f"Selected alpha: {alpha}")
            #alpha = np.zeros(3)  # 测试阶段不使用RL输出的alpha，而是固定为0
            alpha_Q = alpha[:2]           # (2,)
            alpha_R = alpha[2:].reshape(-1)  # (1,)
            action_mpc = mpc_controller.solve(obs, r_ref, -1, alpha_Q, alpha_R) # MPC用的是原始数据
            writer.add_scalar(f"Action_test", action_mpc, env.t_current*1000)

        next_obs, reward, done, info = env.step(action_mpc, alpha, inner_steps_per_control)
        obs = next_obs
        state = list(obs[:3])
        state = Normalize(state)
        episode_return += reward

        K += 1
        if K > 10:
            K = 0
    print(f"Test episode - Return: {episode_return:.3f}")
    
def train(env, mpc_controller, agent, num_episodes, inner_steps_per_control=1):
    writer = SummaryWriter(f"runs/ppo_tireburst_{time.strftime('%Y%m%d-%H%M%S')}")
    return_list = []

    for ep in range(num_episodes):
        obs = env.reset() # 用于仿真迭代
        state = list(obs[:3]) # 用于训练RL
        state = Normalize(state)

        K = 10
        action_mpc = 0 # 初始化动作, RL返回的是nparray
        alpha = np.zeros(3) # 初始化alpha, MPC用的是原始数据
        episode_return = K_step_reward = 0
        r_ref = np.zeros((2,mpc_controller.N))
        done = False 
        transition_dict = {'states': [],'actions': [],'next_states': [],'rewards': [],'dones': [],'log_probs': []}

        while not done:
            if obs[5] != 0: 
                if K == 10:  # 每隔1个时间步选择一次动作
                    decision_state = state.copy()
                    alpha, log_prob = agent.select_action(state)
                    # print(alpha)
                    alpha = np.array(alpha).reshape(-1)
                    K_step_reward = 0

                alpha_Q = alpha[:2]           # (2,)
                alpha_R = alpha[2:].reshape(-1)  # (1,)
                action_mpc = mpc_controller.solve(obs, r_ref, -1, alpha_Q, alpha_R) # MPC用的是原始数据
                writer.add_scalar(f"Action/episode_{ep}", action_mpc, env.t_current*1000)

            next_obs, reward, done, info = env.step(action_mpc, alpha, inner_steps_per_control)
            
            next_state = list(next_obs[:3])
            next_state = Normalize(next_state)
            K_step_reward += reward
            episode_return += reward 


            if obs[5] != 0 and K == 10:  # 在动作更新的时间步记录transition
                transition_dict['states'].append(decision_state)
                transition_dict['actions'].append(alpha)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(K_step_reward)
                transition_dict['dones'].append(done)
                transition_dict['log_probs'].append(log_prob)

            obs = next_obs
            state = next_state
            
            K += 1
            if K > 10:
                K = 0

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
        state_dim=3, action_dim=3, hidden_dim=256,
        actor_lr=args.actor_learning_rate, critic_lr=args.critic_learning_rate,
        gamma=args.gamma, lmbda=args.lmbda, epochs=args.K_epochs, eps=args.eps_clip,
        device=torch.device("cuda") if torch.cuda.is_available() else "cpu")

    inner_steps_per_control = 20

    if args.mode == "test":
        agent.reload_model('ppo_checkpoint_episode2.pth')
        test(env, mpc_controller, agent, inner_steps_per_control) 
        
    elif args.mode == "train":
        train(env, mpc_controller, agent, args.max_episodes, inner_steps_per_control)
        agent.save_model('ppo_checkpoint_episode2.pth', episode=args.max_episodes)
