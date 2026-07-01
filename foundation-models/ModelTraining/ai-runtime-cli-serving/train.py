"""Train a PyTorch MLP on the AI Runtime GPU and register it to Unity Catalog.

This is the serving-ready sibling of ``ai-runtime-cli-intro/train.py``. The only
substantive difference is that this script logs the model *with a signature and
input example* and *registers it to Unity Catalog*, both of which Model Serving
requires. Everything is parameterized via CLI flags so the same script serves
any hyperparameters / model name (override them in the ``command:`` of train.yaml).

    command: python $CODE_SOURCE_PATH/train.py \
      --registered-model-name custom_ml.models.air_mlp_classifier --epochs 20
"""

import argparse

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from mlflow.models import infer_signature
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Where to register the served model (3-level Unity Catalog name).
    p.add_argument("--registered-model-name", required=True,
                   help="Unity Catalog name: <catalog>.<schema>.<model>")
    # Hyperparameters — all optional with sensible defaults.
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

    # --- Serving-ready logging ---------------------------------------------
    # Move the model to CPU BEFORE logging. With serialization_format="pickle"
    # (torch.save of the whole module), a model saved while on the GPU keeps
    # CUDA tensors, which then fail to deserialize inside a CPU serving container
    # ("Attempting to deserialize object on a CUDA device ..."). That failure
    # surfaces as an endpoint stuck NOT_READY/pending forever. Logging from CPU
    # makes the artifact portable to CPU (or GPU) serving.
    model.to("cpu").eval()

    # A signature + input example are REQUIRED for Unity Catalog registration
    # and Model Serving. Build them from a small CPU batch so the schema matches
    # what the serving container receives: float32, shape [-1, n_features].
    example = np.random.default_rng(args.seed).standard_normal(
        (5, args.n_features)
    ).astype(np.float32)
    with torch.no_grad():
        example_out = model(torch.from_numpy(example)).numpy()
    signature = infer_signature(example, example_out)

    # Register straight into Unity Catalog. 'pickle' (classic torch.save) avoids
    # the pt2 TensorSpec path; the signature above supplies serving validation.
    mlflow.set_registry_uri("databricks-uc")
    info = mlflow.pytorch.log_model(
        model,
        name="model",
        signature=signature,
        input_example=example,
        serialization_format="pickle",
        registered_model_name=args.registered_model_name,
    )
    mlflow.end_run()

    print(f"Registered {args.registered_model_name} "
          f"version {info.registered_model_version}")
    print("Training complete; model logged and registered to Unity Catalog.")


if __name__ == "__main__":
    main()
