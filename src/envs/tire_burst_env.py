from pathlib import Path

import numpy as np

from src.envs.carsim_env import CarSimEnv
from src.models.reward_model import compute_reward
from src.utils.utils import modify_run_all_par, replace_tire_force_tables


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BURST_SEVERITY_LEVELS = (0.75, 0.50, 0.25)
BURST_TIRE_CODES = (0, 1, 2, 3)
BURST_TIME = 2  # seconds after which the burst is applied

class TireBurstEnv(CarSimEnv): 

    def __init__(self, sim_path: str):
        super().__init__(sim_path)
        self.v_min = 70
        self.v_max = 160
        self.mu_min = 0.3
        self.mu_max = 1.0
        self.curriculum_stage = 0
        self.previous_action = None
        self.burst_table_root = PROJECT_ROOT / "generated_tire_tables"
        self.burst_tire_par_files = {
            "Fx": Path(
                r"D:\CarSim2023\Databases\Tires\Fx\TireFx_59ff8167-"
                r"24e0-484d-845e-6946e5597b34.par"
            ),
            "Fy": Path(
                r"D:\CarSim2023\Databases\Tires\Fy\TireFy_efd067b1-"
                r"5b7f-406a-830a-324252abc172.par"
            ),
            "Mz": Path(
                r"D:\CarSim2023\Databases\Tires\Mz\TireMz_c3c085d8-"
                r"b8db-4f26-a6e4-032014e75fa1.par"
            ),
        }

    def reset(self, ep):
        par_path = r"D:\CarSim2023\Databases\Results\Run_60366987-856d-45b1-b359-7939e85af520\Run_all.par"

        # Focus most samples near the current difficulty boundary. A small
        # replay fraction keeps the policy from forgetting easier conditions.
        self.curriculum_stage = int(np.log2( (ep+300) // 300) +1)
        self.v_max = min(150.0, self.v_min + 20.0 * self.curriculum_stage)
        self.mu_min = max(0.3, self.mu_max - 0.15 * self.curriculum_stage)
        self.v = np.random.uniform(self.v_min, self.v_max)
        self.mu = np.random.uniform(self.mu_min, self.mu_max)
        self.m = np.random.uniform(1500, 2000)
        self.Iz = self.m * 1.21

        normalized_mu = (self.mu - 0.65) / 0.35
        normalized_m = (self.m - 1500.0) / 500.0

        self.burst_tire = ep % 4
        self.burst_severity = float(np.random.choice(BURST_SEVERITY_LEVELS[:self.curriculum_stage]))
        self.burst_active = False
        self.burst_info = np.zeros(4)
        self.burst_info[self.burst_tire] = self.burst_severity

        self.env_info = np.concatenate([self.burst_info, np.array([normalized_mu, normalized_m], dtype=np.float32)])
        self.target_vx = 0.0
        print(f"Resetting environment with burst_tire={self.burst_tire}, burst_severity={self.burst_severity}")
        print(f"Resetting environment with v={self.v}, mu={self.mu}, m={self.m}, Iz={self.Iz}")

        # 这里做domain randomization，直接在环境层面设置 VS 的参数
        modify_run_all_par(self.v, self.mu, self.m, self.Iz, par_path)
        severity_dir = self.burst_table_root / (
            f"burst_{round(100 * self.burst_severity):03d}"
        )
        replace_tire_force_tables(
            severity_dir,
            self.burst_tire_par_files,
        )

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
        return initial_export, self.env_info
    
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

        # Keep the activation state fixed for this complete transition.  When
        # the pre-burst integration first reaches BURST_TIME, capture its final
        # speed and activate the burst for the *next* policy decision.
        burst_was_active = self.burst_active
        carsim_inputs[5] = self.burst_tire if burst_was_active else 4

        export_vars, done, info = super().step(carsim_inputs, inner_steps)

        if not self.burst_active and self.t_current >= BURST_TIME:
            self.target_vx = float(export_vars[0])
            self.burst_active = True

        reward, reward_components = compute_reward(
            export_vars,
            action,
            self.previous_action,
            burst_was_active,
            self.target_vx,
            info
        )

        self.previous_action = action.copy()

        info.update(
            {f"reward_{name}": value for name, value in reward_components.items()}
        )
        info["burst_active"] = burst_was_active
        info["target_vx"] = self.target_vx

        return export_vars, self.env_info, reward, done, info
    
    def close(self):
        super().close()
