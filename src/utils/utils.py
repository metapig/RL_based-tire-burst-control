import random
import re
import numpy as np
import torch

def set_seed(seed=259):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def Normalize(state):
    training_state = [0.0] * 3
    training_state[0] = state[0] / 70
    training_state[1] = state[1] / 0.4 
    training_state[2] = state[2] / 4
    return training_state

def build_policy_state(obs, env_feature):
    """Build PPO input from normalized vehicle state and environment feature."""
    if np.isscalar(env_feature):
        env_values = [float(env_feature)]
    else:
        env_values = np.asarray(env_feature, dtype=np.float32).reshape(-1).tolist()
    return Normalize(list(obs[:3])) + env_values

def compute_advantage(gamma, lmbda, td_delta):

    advantages = []
    advantage = 0.0

    for delta in reversed(td_delta):

        advantage = gamma * lmbda * advantage + delta
        advantages.append(advantage)

    advantages.reverse()

    return torch.stack(advantages).squeeze()

def modify_run_all_par(v, mu, file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        # 修改初速度
        if re.match(r"\s*SV_VXS", line):
            new_lines.append(f"SV_VXS {v}\n")
        
        # 修改路面附着
        elif re.match(r"\s*MU_ROAD_CONSTANT", line):
            new_lines.append(f"MU_ROAD_CONSTANT {mu}\n")
        
        else:
            new_lines.append(line)

    with open(file_path, "w") as f:
        f.writelines(new_lines)
