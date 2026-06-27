import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.mlp_env_encoder import EncoderDataset, MLPEnvEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MLPEnvEncoder on saved encoder datasets.")
    parser.add_argument("--data", nargs="+", required=True, help="One or more .npz dataset files")
    parser.add_argument("--checkpoint", required=True, help="Encoder checkpoint from train_env_encoder.py")
    parser.add_argument("--batch_size", type=int, default=512)
    return parser.parse_args()


def segment_mae(preds, labels, mask):
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(preds[mask] - labels[mask])))


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    dataset = EncoderDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = MLPEnvEncoder.from_checkpoint(args.checkpoint, map_location=device).to(device)
    model.eval()

    preds = []
    labels = []
    total_mse = 0.0
    total_count = 0
    for histories, mu in loader:
        histories = histories.to(device)
        mu = mu.to(device)
        pred = model(histories)
        total_mse += F.mse_loss(pred, mu, reduction="sum").item()
        total_count += len(mu)
        preds.append(pred.cpu().numpy())
        labels.append(mu.cpu().numpy())

    preds = np.concatenate(preds, axis=0).reshape(-1)
    labels = np.concatenate(labels, axis=0).reshape(-1)
    mse = total_mse / max(1, total_count)
    mae = float(np.mean(np.abs(preds - labels)))
    low = labels < 0.4
    mid = (labels >= 0.4) & (labels <= 0.7)
    high = labels > 0.7

    print(f"samples: {total_count}")
    print(f"mse: {mse:.6f}")
    print(f"mae: {mae:.6f}")
    print(f"mae_mu_lt_0.4: {segment_mae(preds, labels, low):.6f}")
    print(f"mae_mu_0.4_to_0.7: {segment_mae(preds, labels, mid):.6f}")
    print(f"mae_mu_gt_0.7: {segment_mae(preds, labels, high):.6f}")


if __name__ == "__main__":
    main()
