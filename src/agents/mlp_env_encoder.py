from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MLPEnvEncoder(nn.Module):
    """MLP encoder that predicts road friction mu from flattened history."""

    def __init__(
        self,
        history_len: int = 25,
        feature_dim: int = 6,
        hidden_dims: Sequence[int] = (128, 128, 64),
        mu_range: Tuple[float, float] = (0.2, 1.0),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.history_len = int(history_len)
        self.feature_dim = int(feature_dim)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.dropout = float(dropout)
        self.mu_min, self.mu_max = float(mu_range[0]), float(mu_range[1])

        input_dim = self.history_len * self.feature_dim
        layers = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() == 2:
            flat = history
        elif history.dim() == 3:
            flat = history.reshape(history.shape[0], -1)
        else:
            raise ValueError("history must have shape (batch, T*F) or (batch, T, F).")

        expected_dim = self.history_len * self.feature_dim
        if flat.shape[-1] != expected_dim:
            raise ValueError(f"Expected flattened history dim {expected_dim}, got {flat.shape[-1]}.")

        mu_unit = self.net(flat)
        return self.mu_min + mu_unit * (self.mu_max - self.mu_min)

    @torch.no_grad()
    def predict_mu(self, history: np.ndarray, device: Optional[Union[torch.device, str]] = None) -> float:
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        history_tensor = torch.as_tensor(history, dtype=torch.float32, device=device)
        if history_tensor.dim() == 2:
            history_tensor = history_tensor.unsqueeze(0)
        return float(self(history_tensor).squeeze().detach().cpu().item())

    def checkpoint_payload(self) -> dict:
        return {
            "model_state_dict": self.state_dict(),
            "history_len": self.history_len,
            "feature_dim": self.feature_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "mu_range": (self.mu_min, self.mu_max),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Union[str, Path], map_location: Union[str, torch.device] = "cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model = cls(
            history_len=checkpoint.get("history_len", 25),
            feature_dim=checkpoint.get("feature_dim", 6),
            hidden_dims=tuple(checkpoint.get("hidden_dims", (128, 128, 64))),
            dropout=float(checkpoint.get("dropout", 0.0)),
            mu_range=tuple(checkpoint.get("mu_range", (0.2, 1.0))),
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        return model


class EncoderDataset(Dataset):
    """Dataset for supervised mu prediction from saved history windows."""

    def __init__(self, files: Union[str, Path, Iterable[Union[str, Path]]]):
        if isinstance(files, (str, Path)):
            files = [files]

        histories = []
        labels = []
        for file in files:
            data = np.load(file)
            if "histories" not in data:
                raise ValueError(f"{file} does not contain 'histories'.")
            label_key = "labels" if "labels" in data else "mu"
            if label_key not in data:
                raise ValueError(f"{file} does not contain 'labels' or 'mu'.")
            histories.append(np.asarray(data["histories"], dtype=np.float32))
            labels.append(np.asarray(data[label_key], dtype=np.float32).reshape(-1, 1))

        self.histories = np.concatenate(histories, axis=0)
        self.labels = np.concatenate(labels, axis=0)
        if len(self.histories) != len(self.labels):
            raise ValueError("histories and labels must have the same first dimension.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.histories[idx]).float(),
            torch.from_numpy(self.labels[idx]).float(),
        )
