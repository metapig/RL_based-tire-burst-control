import numpy as np

from src.models.parameters import ACTION_SCALES


# CarSim export order and units:
# VX [km/h], VY [km/h], AVZ [deg/s], YO [m], YAW [deg].
VEHICLE_STATE_SCALES = np.asarray(
    (100.0, 5.0, 10.0, 5.0, 15.0), dtype=np.float64
)
MU_CENTER = 0.65
MU_HALF_RANGE = 0.35
VEHICLE_STATE_DIM = 5
PREVIOUS_ACTION_DIM = 5
STATE_DIM = VEHICLE_STATE_DIM + 1 + PREVIOUS_ACTION_DIM


def build_state(export_vars, previous_action):
    """Build the Markov state used by both actor and critic.

    The previous action is part of the state because the reward contains an
    action-increment penalty. The current road friction is read from CarSim's
    INIT_MU export (index 7), rather than duplicated from Python-side sampling.
    """
    export_vars = np.asarray(export_vars, dtype=np.float64).reshape(-1)
    previous_action = np.asarray(previous_action, dtype=np.float64).reshape(-1)

    if export_vars.size < 8:
        raise ValueError(
            f"Expected at least 8 CarSim exports, got {export_vars.size}."
        )
    if previous_action.shape != (PREVIOUS_ACTION_DIM,):
        raise ValueError(
            "Expected previous_action with shape "
            f"({PREVIOUS_ACTION_DIM},), got {previous_action.shape}."
        )
    normalized_vehicle_state = (
        export_vars[:VEHICLE_STATE_DIM] / VEHICLE_STATE_SCALES
    )
    normalized_mu = (export_vars[7] - MU_CENTER) / MU_HALF_RANGE
    normalized_previous_action = previous_action / np.asarray(
        ACTION_SCALES, dtype=np.float64
    )
    state = np.concatenate(
        (
            normalized_vehicle_state,
            np.asarray((normalized_mu,)),
            normalized_previous_action,
        )
    ).astype(np.float32)

    if not np.all(np.isfinite(state)):
        raise ValueError("State contains NaN or infinite values.")
    return state
