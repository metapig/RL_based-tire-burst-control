import numpy as np
from src.envs.carsim_env import CarSimEnv
from src.models.reward_model import compute_reward
from src.utils.utils import modify_run_all_par

class TireBurstEnv(CarSimEnv): 

    def __init__(self, sim_path: str):
        super().__init__(sim_path)
        self.v_min = 70
        self.v_max = 70
        self.mu_min = 1.0
        self.mu_max = 1.0
        self.curriculum_stage = 0
        self.previous_action = None

    def reset(self, ep):
        par_path = r"D:\CarSim2023\Databases\Results\Run_60366987-856d-45b1-b359-7939e85af520\Run_all.par"
        self.burst_tire = -1 #random.choice([-2, -1, 1, 2])

        # Focus most samples near the current difficulty boundary. A small
        # replay fraction keeps the policy from forgetting easier conditions.
        self.curriculum_stage = min(ep // 250, 8)
        self.v_max = min(150.0, self.v_min + 10.0 * self.curriculum_stage)
        self.mu_min = max(0.3, self.mu_max - 0.1 * self.curriculum_stage)
        replay_easier_condition = (
            self.curriculum_stage > 0 and np.random.random() < 0.2
        )

        if replay_easier_condition:
            self.v = np.random.uniform(self.v_min, self.v_max)
            self.mu = np.random.uniform(self.mu_min, self.mu_max)
        else:
            current_v_min = max(self.v_min, self.v_max - 10.0)
            current_mu_max = min(self.mu_max, self.mu_min + 0.1)
            self.v = np.random.uniform(current_v_min, self.v_max)
            self.mu = np.random.uniform(self.mu_min, current_mu_max)
        # self.m = np.random.uniform(1000, 2000)
        # self.Iz = np.random.uniform(2000, 4000)

        # 这里做domain randomization，直接在环境层面设置 VS 的参数
        modify_run_all_par(self.v, self.mu, par_path)

        self.previous_action = None
        initial_export = super().reset()
        if len(initial_export) < 8:
            self.close()
            raise RuntimeError(
                f"Expected at least 8 CarSim exports, got {len(initial_export)}."
            )

        reported_mu = float(initial_export[7])
        if not np.isclose(reported_mu, self.mu, rtol=0.0, atol=1e-5):
            self.close()
            raise RuntimeError(
                "CarSim friction mismatch: Python requested "
                f"mu={self.mu:.6f}, but INIT_MU exported {reported_mu:.6f}."
            )
        return initial_export
    
    def step(self, action, inner_steps=1):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (5,):
            raise ValueError(f"Expected a 5-D action, got shape {action.shape}.")
        if self.n_import != 6:
            raise RuntimeError(
                f"Expected 6 CarSim import channels, got {self.n_import}."
            )

        carsim_inputs = np.zeros(6, dtype=np.float64)
        carsim_inputs[:5] = action
        carsim_inputs[5] = self.burst_tire if self.t_current >= 2 else 0

        export_vars, done, info = super().step(carsim_inputs, inner_steps)
        reward, reward_components = compute_reward(
            export_vars,
            action,
            self.previous_action,
        )
        self.previous_action = action.copy()
        info.update(
            {f"reward_{name}": value for name, value in reward_components.items()}
        )

        return export_vars, reward, done, info
    
    def close(self):
        super().close()
