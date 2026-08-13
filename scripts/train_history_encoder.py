import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.agents.history_encoder import AdaptationDataset, CNN


def train(net, dataset, device, num_epochs=10, batch_size=32,
          learning_rate=1e-3):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    net.train()
    for epoch in range(num_epochs):
        loss_sum = 0.0
        for history_buffer, target_z in dataloader:
            history_buffer = history_buffer.to(device)
            target_z = target_z.to(device)
            predicted_z = net(history_buffer)
            loss = loss_fn(predicted_z, target_z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(history_buffer)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Loss: {loss_sum / len(dataset):.6f}"
        )

def test(history_encoder, dataset, device):
    dataloder = DataLoader(dataset, batch_size=32, shuffle=False)
    loss_fn = torch.nn.MSELoss()
    history_encoder.eval()

    total_loss = 0.0
    with torch.no_grad():
        for history_buffer, target_z in dataloder:
            history_buffer = history_buffer.to(device)
            target_z = target_z.to(device)
            predicted_z = history_encoder(history_buffer)
            loss = loss_fn(predicted_z, target_z)
            total_loss += loss.item() * len(history_buffer)

    print(f"Test Loss: {total_loss / len(dataset):.6f}")
    return total_loss / len(dataset)

def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="adaptation_dataset.pt")
    parser.add_argument("--checkpoint", default="history_encoder.pth")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--test", action="store_true", help="Run in test mode (no training)")
    parser.add_argument("--train", action="store_true", help="Run in training mode (train the model)")
    return parser.parse_args()


if __name__ == "__main__":
    args = initialize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AdaptationDataset.load(args.dataset)
    history_encoder = CNN(input_channels=11, output_channels=8).to(device)

    checkpoint = Path(args.checkpoint)

    if args.test:
        history_encoder.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        test(history_encoder, dataset, device)

    if args.resume and checkpoint.exists():
        history_encoder.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        print(f"Resuming from checkpoint: {checkpoint}")

    if args.train or args.resume:
        train(
            history_encoder,
            dataset,
            device,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(history_encoder.state_dict(), checkpoint)
        print(f"Saved history encoder to {checkpoint}")
        
