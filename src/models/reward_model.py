import numpy as np

def compute_reward(state, action_mpc, alpha):
    """
    自动记忆上一时刻 action 和 alpha
    """
    vy, r, y, psi = state[1:5]

    vy_norm = vy / 0.1
    r_norm = r / 10
    y_norm = y / 10
    psi_norm = psi / 10
    action_norm = action_mpc / 1000

    if not hasattr(compute_reward, "prev_alpha"):
        compute_reward.prev_alpha = None

    r_track = - 1.0 * y_norm**2 - 1.0 * psi_norm**2
    r_stability = - 1.0 * r_norm**2 - 1.0 * vy_norm**2

    # print(f"Normalized: y_norm={y_norm:.3f}, psi_norm={psi_norm:.3f}, r_norm={r_norm:.3f}, vy_norm={vy_norm:.3f}")

    if compute_reward.prev_alpha is None:
        r_param_smooth = 0
    else:
        r_param_smooth = -np.sum((alpha - compute_reward.prev_alpha)**2)

    reward = r_track + r_stability + r_param_smooth
    # print(f"Reward components: track={r_track:.3f}, stability={r_stability:.3f}, control_smooth={r_control_smooth:.3f}, param_smooth={r_param_smooth:.3f}, total={reward:.3f}")

    compute_reward.prev_alpha = alpha.copy()

    return reward