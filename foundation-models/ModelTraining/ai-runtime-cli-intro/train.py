"""Minimal PyTorch GPU training example for the Databricks AI Runtime CLI.

Trains a tiny MLP on a synthetic binary-classification problem, runs on the
GPU provisioned by `air`, and logs params/metrics/model to the MLflow
experiment that the AI Runtime creates for the run.

All hyperparameters are CLI args, so you can override them from train.yaml, e.g.
    command: python $CODE_SOURCE_PATH/train.py --epochs 20 --lr 5e-4
"""

import argparse

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=8192)
    p.add_argument("--n-features", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def make_data(args: argparse.Namespace, device: torch.device) -> DataLoader:
    """Synthetic dataset: label = 1 when a linear combination of features > 0."""
    torch.manual_seed(args.seed)
    x = torch.randn(args.n_samples, args.n_features)
    w = torch.randn(args.n_features, 1)
    logits = x @ w + 0.1 * torch.randn(args.n_samples, 1)
    y = (logits > 0).float()
    ds = TensorDataset(x.to(device), y.to(device))
    return DataLoader(ds, batch_size=args.batch_size, shuffle=True)


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    loader = make_data(args, device)
    model = MLP(args.n_features, args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # `air` starts the MLflow run; log directly to the active run.
    mlflow.log_params({**vars(args), "device": device.type})

    for epoch in range(args.epochs):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        avg_loss = epoch_loss / total
        acc = correct / total
        print(f"epoch {epoch + 1:>2}/{args.epochs}  loss={avg_loss:.4f}  acc={acc:.4f}")
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        mlflow.log_metric("train_accuracy", acc, step=epoch)

    # Use 'pickle' serialization (classic torch.save); the default 'pt2' format
    # requires a TensorSpec signature this simple example doesn't provide.
    # Skip input_example to avoid signature/serving validation, which can stall.
    model.eval()
    mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")
    mlflow.end_run()
    print("Training complete; model logged to MLflow.")


if __name__ == "__main__":
    main()
