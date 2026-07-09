"""Step 1 — fine-tune + register.

Fine-tunes a small BERT-family model (DistilBERT by default) for binary
sentiment analysis on a trimmed slice of a built-in Hugging Face dataset
(GLUE SST-2 by default), running on the GPU that `air` provisions. Params and
metrics go to MLflow; the model is logged as a `text-classification` pipeline
and registered to Unity Catalog so step 2 (`deploy.py`) can serve it.

The registered version is tagged with an MLflow registry alias (--alias /
MODEL_ALIAS) so the deploy step picks up exactly this version — no "latest
version" guessing that could race with a concurrent run.

The dataset is trimmed (--train-size / --eval-size) and capped at one epoch by
default so a full fine-tune finishes in a couple of minutes on a single A10.
"""

import argparse
import os

import mlflow
import numpy as np
import torch
from datasets import load_dataset
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

# SST-2: 0 = negative, 1 = positive. Baked into the model config so the served
# pipeline returns human-readable labels instead of LABEL_0 / LABEL_1.
ID2LABEL = {0: "negative", 1: "positive"}
LABEL2ID = {"negative": 0, "positive": 1}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name",
                   default=os.environ.get("MODEL_NAME", "main.default.distilbert_sentiment"),
                   help="Unity Catalog model name: <catalog>.<schema>.<model>")
    p.add_argument("--alias", default=os.environ.get("MODEL_ALIAS", "champion"),
                   help="Registry alias assigned to this version for the deploy step to resolve")
    p.add_argument("--base-model", default=os.environ.get("BASE_MODEL", "distilbert-base-uncased"),
                   help="Hugging Face model id to fine-tune")
    p.add_argument("--dataset", default=os.environ.get("DATASET", "glue"))
    p.add_argument("--dataset-config", default=os.environ.get("DATASET_CONFIG", "sst2"))
    p.add_argument("--text-column", default=os.environ.get("TEXT_COLUMN", "sentence"))
    p.add_argument("--train-size", type=int, default=2048,
                   help="Number of training rows to keep (trim for speed)")
    p.add_argument("--eval-size", type=int, default=512)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_splits(args: argparse.Namespace):
    """Load, shuffle, and trim the train/validation splits."""
    ds = load_dataset(args.dataset, args.dataset_config)
    train = ds["train"].shuffle(seed=args.seed).select(range(min(args.train_size, ds["train"].num_rows)))
    eval_ = ds["validation"].shuffle(seed=args.seed).select(range(min(args.eval_size, ds["validation"].num_rows)))
    return train, eval_


def main() -> None:
    args = parse_args()
    mlflow.set_registry_uri("databricks-uc")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_ds, eval_ds = load_splits(args)
    print(f"Fine-tuning {args.base_model} on {args.dataset}/{args.dataset_config}: "
          f"{train_ds.num_rows} train / {eval_ds.num_rows} eval rows")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(batch[args.text_column], truncation=True, max_length=args.max_length)

    train_tok = train_ds.map(tokenize, batched=True)
    eval_tok = eval_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(device)

    training_args = TrainingArguments(
        output_dir="/tmp/hf-sentiment",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        report_to=[],            # we log to MLflow ourselves; avoid the HF autologger
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        tokenizer=tokenizer,
    )

    mlflow.log_params({
        "base_model": args.base_model,
        "dataset": f"{args.dataset}/{args.dataset_config}",
        "train_size": train_ds.num_rows,
        "eval_size": eval_ds.num_rows,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "device": device.type,
    })

    train_result = trainer.train()
    mlflow.log_metric("train_loss", train_result.training_loss)

    # Manual accuracy on the eval split — avoids version-sensitive
    # evaluation_strategy / eval_strategy TrainingArguments plumbing.
    preds = trainer.predict(eval_tok)
    y_pred = np.argmax(preds.predictions, axis=1)
    y_true = np.array(eval_tok["label"])
    accuracy = float((y_pred == y_true).mean())
    print(f"train_loss={train_result.training_loss:.4f}  eval_accuracy={accuracy:.4f}")
    mlflow.log_metric("eval_accuracy", accuracy)

    # Move to CPU before logging so the CPU serving endpoint loads it without a
    # GPU dependency (device=-1 keeps the pipeline off CUDA).
    model.cpu()
    sentiment = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)

    input_example = ["I loved this movie, it was fantastic!", "Worst experience ever, do not recommend."]
    signature = infer_signature(input_example, sentiment(input_example))

    model_info = mlflow.transformers.log_model(
        transformers_model=sentiment,
        name="model",
        task="text-classification",
        signature=signature,
        input_example=input_example,
        registered_model_name=args.model_name,
    )
    mlflow.end_run()

    # Hand the exact version off to the deploy step via a registry alias.
    version = model_info.registered_model_version
    MlflowClient(registry_uri="databricks-uc").set_registered_model_alias(
        name=args.model_name, alias=args.alias, version=version
    )
    print(f"Training complete; registered {args.model_name} v{version} as @{args.alias}.")


if __name__ == "__main__":
    main()
