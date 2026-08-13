import numpy as np

from src.models.parameters import ACTION_SCALES


REWARD_SCALE = 0.1


def _smooth_l1(value):
    """Quadratic near zero and linear for large errors."""
    absolute = np.abs(value)
    return np.where(absolute < 1.0, 0.5 * value**2, absolute - 0.5)


def compute_reward(state, action, previous_action, burst_active, target_vx, info):
    """Return a robust state/control reward and its individual components."""
    vx, vy, yaw_rate, lateral_position, yaw_angle = np.asarray(
        state[:5], dtype=np.float64
    )
    vx_target = float(target_vx)
    action = np.asarray(action, dtype=np.float64)
    action_scales = np.asarray(ACTION_SCALES, dtype=np.float64)

    # CarSim units: km/h, deg/s, m, deg.
    speed_error = (vx - vx_target) / 10.0
    vy_error = vy
    yaw_rate_error = yaw_rate / 2.0 # 5.0
    lateral_error = lateral_position / 2.0 # 5.0
    yaw_error = yaw_angle / 2.0 # 15

    speed_cost =  _smooth_l1(speed_error)

    track_cost = (
        4 * _smooth_l1(lateral_error)
        + 3 * _smooth_l1(yaw_error)
    )
    stability_cost = (
        _smooth_l1(vy_error)
        + _smooth_l1(yaw_rate_error)
    )

    normalized_action = action / action_scales
    control_cost = 0.5 * (
        normalized_action[0] ** 2
        + 0.5 *np.sum(normalized_action[1:] ** 2)
    )

    smoothness_cost = 0.0
    if previous_action is not None:
        normalized_delta = (
            action - np.asarray(previous_action, dtype=np.float64)
        ) / action_scales
        smoothness_cost = (
            normalized_delta[0] ** 2
            + 0.5 * np.sum(normalized_delta[1:] ** 2)
        )

    terminal_penalty = 0.0
    if info.get("terminated", False):
        terminal_penalty = -500.0

    components = {
        "speed": -REWARD_SCALE * float(speed_cost),
        "track": -REWARD_SCALE * float(track_cost),
        "stability": -REWARD_SCALE * float(stability_cost),
        "control": -REWARD_SCALE * float(control_cost),
        "smoothness": -REWARD_SCALE * float(smoothness_cost),
        "terminal": terminal_penalty
    }
    if not burst_active:
        components = {name: 0.0 for name in components}

    reward = float(sum(components.values()))
    return reward, components
