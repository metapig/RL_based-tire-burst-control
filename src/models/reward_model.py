import numpy as np

from src.models.parameters import ACTION_SCALES


REWARD_SCALE = 0.1


def _smooth_l1(value):
    """Quadratic near zero and linear for large errors."""
    absolute = np.abs(value)
    return np.where(absolute < 1.0, 0.5 * value**2, absolute - 0.5)


def compute_reward(state, action, previous_action=None):
    """Return a robust state/control reward and its individual components."""
    vy, yaw_rate, lateral_position, yaw_angle = np.asarray(
        state[1:5], dtype=np.float64
    )
    action = np.asarray(action, dtype=np.float64)
    action_scales = np.asarray(ACTION_SCALES, dtype=np.float64)

    # CarSim units: km/h, deg/s, m, deg.
    vy_error = vy / 0.5
    yaw_rate_error = yaw_rate / 5.0
    lateral_error = lateral_position / 5.0
    yaw_error = yaw_angle / 15.0

    track_cost = (
        2.5 * _smooth_l1(lateral_error)
        + 1.5 * _smooth_l1(yaw_error)
    )
    stability_cost = (
        2.0 * _smooth_l1(vy_error)
        + 2.0 * _smooth_l1(yaw_rate_error)
    )

    normalized_action = action / action_scales
    control_cost = 0.01 * (
        normalized_action[0] ** 2
        + 0.25 * np.sum(normalized_action[1:] ** 2)
    )

    smoothness_cost = 0.0
    if previous_action is not None:
        normalized_delta = (
            action - np.asarray(previous_action, dtype=np.float64)
        ) / action_scales
        smoothness_cost = 0.02 * (
            normalized_delta[0] ** 2
            + 0.25 * np.sum(normalized_delta[1:] ** 2)
        )

    components = {
        "track": -REWARD_SCALE * float(track_cost),
        "stability": -REWARD_SCALE * float(stability_cost),
        "control": -REWARD_SCALE * float(control_cost),
        "smoothness": -REWARD_SCALE * float(smoothness_cost),
    }
    reward = float(sum(components.values()))
    return reward, components
