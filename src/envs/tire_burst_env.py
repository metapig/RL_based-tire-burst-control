import numpy as np
import random
from src.envs.carsim_env import CarSimEnv
from src.models.reward_model import compute_reward
from src.utils.utils import modify_run_all_par

class TireBurstEnv(CarSimEnv): 

    def __init__(self, sim_path: str):
        super().__init__(sim_path)

    def reset(self):
        par_path = r"D:\CarSim2023\Databases\Results\Run_60366987-856d-45b1-b359-7939e85af520\Run_all.par"
        self.burst_tire = -1#random.choice([-2, -1, 1, 2])
        self.v = np.random.uniform(70, 100)
        self.mu = np.random.uniform(0.2, 1.0)
        # self.m = np.random.uniform(1000, 2000)
        # self.Iz = np.random.uniform(2000, 4000)

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