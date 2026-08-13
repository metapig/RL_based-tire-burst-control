from collections import deque
from pathlib import Path

import numpy as np
import torch


HISTORY_LENGTH = 25


class CNN(torch.nn.Module):
    def __init__(self, input_channels=11, output_channels=8):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(input_channels, 32, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_channels),
        )

    def forward(self, x):
        return self.net(x)


class HistoryBuffer:
    def __init__(self, maxlen=HISTORY_LENGTH):
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)

    def reset(self, initial_state):
        """Use the initial state to pad the causal history to full length."""
        initial_state = np.asarray(initial_state, dtype=np.float32)
        self.buffer.clear()
        for _ in range(self.maxlen):
            self.buffer.append(initial_state.copy())

    def append(self, item):
        self.buffer.append(np.asarray(item, dtype=np.float32).copy())

    def as_array(self):
        return np.stack(self.buffer, axis=0).T.astype(np.float32)

    def __len__(self):
        return len(self.buffer)


class AdaptationDataset(torch.utils.data.Dataset):
    def __init__(self, history_buffers, target_zs):
        self.history_buffers = torch.as_tensor(
            np.asarray(history_buffers), dtype=torch.float32
        )
        self.target_zs = torch.as_tensor(
            np.asarray(target_zs), dtype=torch.float32
        )

    def __len__(self):
        return len(self.history_buffers)

    def __getitem__(self, index):
        return self.history_buffers[index], self.target_zs[index]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "history_buffers": self.history_buffers,
                "target_zs": self.target_zs,
            },
            path,
        )

    @classmethod
    def load(cls, path):
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            data = torch.load(path, map_location="cpu")
        return cls(data["history_buffers"], data["target_zs"])
