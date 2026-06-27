from collections import deque
from typing import Iterable, Sequence

import numpy as np


HISTORY_FEATURE_NAMES = (
    "vx",
    "vy",
    "r",
    "y",
    "psi",
    "action_mpc",
)

# Keep the encoder input numerically compact. These scales follow the
# project's current PPO/reward normalization conventions where available.
HISTORY_FEATURE_SCALES = np.array(
    [70.0, 0.4, 4.0, 10.0, 10.0, 1000.0],
    dtype=np.float32,
)


def build_history_row(
    obs: Sequence[float],
    action_mpc: float,
    *,
    normalize: bool = True,
) -> np.ndarray:
    
    """Build one encoder history row from CarSim observation and controller inputs."""
    obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if obs_arr.size < 5:
        raise ValueError(f"Expected obs with at least 5 values, got {obs_arr.size}.")

    row = np.array(
        [
            obs_arr[0],
            obs_arr[1],
            obs_arr[2],
            obs_arr[3],
            obs_arr[4],
            float(action_mpc)
        ],
        dtype=np.float32,
    )
    if normalize:
        row = row / HISTORY_FEATURE_SCALES
    return row


class HistoryBuffer:
    """Fixed-length buffer for RMA-style state/action histories."""

    def __init__(
        self,
        history_len: int = 25,
        feature_dim: int = len(HISTORY_FEATURE_NAMES),
        *,
        pad_mode: str = "repeat_first",
    ):
        if history_len <= 0:
            raise ValueError("history_len must be positive.")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if pad_mode not in {"repeat_first", "zeros"}:
            raise ValueError("pad_mode must be 'repeat_first' or 'zeros'.")

        self.history_len = int(history_len)
        self.feature_dim = int(feature_dim)
        self.pad_mode = pad_mode
        self._rows = deque(maxlen=self.history_len)

    def clear(self) -> None:
        self._rows.clear()

    def append_row(self, row: Sequence[float]) -> None:
        row_arr = np.asarray(row, dtype=np.float32).reshape(-1)
        if row_arr.size != self.feature_dim:
            raise ValueError(f"Expected row dim {self.feature_dim}, got {row_arr.size}.")
        self._rows.append(row_arr)

    def append(
        self,
        obs: Sequence[float],
        action_mpc: float,
        *,
        normalize: bool = True,
    ) -> None:
        self.append_row(build_history_row(obs, action_mpc, normalize=normalize))

    def as_array(self) -> np.ndarray:
        rows = list(self._rows)
        if not rows:
            return np.zeros((self.history_len, self.feature_dim), dtype=np.float32)

        missing = self.history_len - len(rows)
        if missing <= 0:
            return np.stack(rows[-self.history_len :], axis=0).astype(np.float32)

        if self.pad_mode == "repeat_first":
            pad_row = rows[0]
        else:
            pad_row = np.zeros(self.feature_dim, dtype=np.float32)
        padding = [pad_row.copy() for _ in range(missing)]
        return np.stack(padding + rows, axis=0).astype(np.float32)

    def __len__(self) -> int:
        return len(self._rows)
