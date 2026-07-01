"""Invoke the deployed Model Serving endpoint with synthetic feature vectors.

Sends `n_features`-dimensional rows (matching what train.py learned on) and
prints the raw logits, sigmoid probabilities, and predicted classes. Fully
parameterized so it works against any endpoint / feature width.

    python invoke.py -p my-workspace
    python invoke.py -p my-workspace --endpoint-name air-mlp-classifier \
      --n-rows 4 --n-features 32 --seed 7
"""

import argparse
import math

from databricks.sdk import WorkspaceClient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-p", "--profile", default="DEFAULT",
                   help="Databricks CLI profile (OAuth) to authenticate with.")
    p.add_argument("--endpoint-name", default="air-mlp-classifier")
    p.add_argument("--n-rows", type=int, default=4,
                   help="How many synthetic rows to score.")
    p.add_argument("--n-features", type=int, default=32,
                   help="Must match the model's input width (train.py --n-features).")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def main() -> None:
    args = parse_args()

    # Build synthetic input without a numpy dependency: a small LCG PRNG turned
    # into standard-normal-ish values via the Box-Muller transform. Deterministic
    # given --seed so runs are reproducible.
    state = args.seed or 1

    def rand() -> float:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return (state + 1) / (0x7FFFFFFF + 2)

    def normal() -> float:
        u1, u2 = rand(), rand()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    rows = [[normal() for _ in range(args.n_features)]
            for _ in range(args.n_rows)]

    w = WorkspaceClient(profile=args.profile)
    print(f"Querying '{args.endpoint_name}' with {args.n_rows} x "
          f"{args.n_features} synthetic rows...")
    resp = w.serving_endpoints.query(name=args.endpoint_name, inputs=rows)

    preds = resp.predictions
    print("\n  row  logit      prob     class")
    print("  ---  ---------  -------  -----")
    for i, p in enumerate(preds):
        logit = p[0] if isinstance(p, (list, tuple)) else p
        prob = sigmoid(logit)
        print(f"  {i:>3}  {logit:>9.4f}  {prob:>6.4f}  {int(prob > 0.5)}")


if __name__ == "__main__":
    main()
