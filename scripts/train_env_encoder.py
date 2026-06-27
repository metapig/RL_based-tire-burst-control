import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.mlp_env_encoder import EncoderDataset, MLPEnvEncoder
from src.utils.utils import set_seed


def parse_hidden_dims(value):
    return tuple(int(item) for item in value.split(",") if item.strip())


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLPEnvEncoder to predict road friction mu.")
    parser.add_argument("--data", nargs="+", required=True, help="One or more .npz files from collect_encoder_data.py")
    parser.add_argument("--output", type=str, default="runs/encoder_checkpoints/mlp_env_encoder.pth")
    parser.add_argument("--history_len", type=int, default=25)
    parser.add_argument("--feature_dim", type=int, default=6)
    parser.add_argument("--hidden_dims", type=parse_hidden_dims, default=(128, 128, 64))
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    preds = []
    labels = []
    for histories, mu in loader:
        histories = histories.to(device)
        mu = mu.to(device)
        pred = model(histories)
        total_loss += F.mse_loss(pred, mu, reduction="sum").item()
        preds.append(pred.detach().cpu().numpy())
        labels.append(mu.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0).reshape(-1)
    labels = np.concatenate(labels, axis=0).reshape(-1)
    mse = total_loss / max(1, len(labels))
    mae = float(np.mean(np.abs(preds - labels)))
    return {"mse": float(mse), "mae": mae, "preds": preds, "labels": labels}


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    dataset = EncoderDataset(args.data)
    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_size else None

    model = MLPEnvEncoder(
        history_len=args.history_len,
        feature_dim=args.feature_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metric = float("inf")
    best_payload = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        for histories, mu in train_loader:
            histories = histories.to(device)
            mu = mu.to(device)
            pred = model(histories)
            loss = F.mse_loss(pred, mu)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(mu)
            running_count += len(mu)

        train_mse = running_loss / max(1, running_count)
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device)
            score = metrics["mae"]
            print(
                f"Epoch {epoch:03d}: train_mse={train_mse:.6f} "
                f"val_mse={metrics['mse']:.6f} val_mae={metrics['mae']:.6f}"
            )
        else:
            score = train_mse
            print(f"Epoch {epoch:03d}: train_mse={train_mse:.6f}")

        if score < best_metric:
            best_metric = score
            best_payload = model.checkpoint_payload()
            best_payload.update({"epoch": epoch, "best_metric": best_metric})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_payload if best_payload is not None else model.checkpoint_payload(), output_path)
    print(f"Saved encoder checkpoint to {output_path}")


if __name__ == "__main__":
    main()
