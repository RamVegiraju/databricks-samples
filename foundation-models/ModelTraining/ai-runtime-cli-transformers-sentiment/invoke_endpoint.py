"""Invoke the deployed sentiment endpoint with sample sentences.

Run locally after the train+deploy job finishes. Sends a batch of sentences to
the Model Serving endpoint and prints the predicted sentiment label and score
for each. Pass your own text with repeated --text flags.

Auth reuses your Databricks CLI profile (the same one passed to `air`).
Requires `pip install databricks-sdk`.
"""

import argparse
import os

from databricks.sdk import WorkspaceClient

DEFAULT_SENTENCES = [
    "I loved this movie, it was fantastic!",
    "The plot was boring and the acting was terrible.",
    "An absolute masterpiece, highly recommend.",
    "Waste of time, I want my money back.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", default=os.environ.get("ENDPOINT_NAME", "air-transformers-sentiment"),
                   help="Serving endpoint name (defaults to ENDPOINT_NAME env var)")
    p.add_argument("-p", "--profile", default=os.environ.get("DATABRICKS_CONFIG_PROFILE", "my-workspace"),
                   help="Databricks CLI profile (defaults to DATABRICKS_CONFIG_PROFILE env var)")
    p.add_argument("--text", action="append", default=None,
                   help="Sentence to classify; repeat for multiple (defaults to a built-in set)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sentences = args.text or DEFAULT_SENTENCES

    w = WorkspaceClient(profile=args.profile)
    resp = w.serving_endpoints.query(name=args.endpoint, inputs=sentences)

    print(f"Endpoint: {args.endpoint}  |  sent {len(sentences)} sentence(s)\n")
    for sentence, pred in zip(sentences, resp.predictions):
        label = pred.get("label", pred) if isinstance(pred, dict) else pred
        score = pred.get("score") if isinstance(pred, dict) else None
        score_str = f"  score={score:.4f}" if isinstance(score, (int, float)) else ""
        print(f"  [{label:>8}]{score_str}  {sentence}")


if __name__ == "__main__":
    main()
