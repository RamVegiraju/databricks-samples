# AI Runtime CLI — Fine-tune a Transformers Model, then Deploy to Model Serving

Fine-tune DistilBERT for sentiment analysis on a serverless GPU with the
**AI Runtime CLI** (`air`), then deploy the trained model to a **Model Serving**
endpoint — as one chained run:

1. `train.py` — fine-tune `distilbert-base-uncased` on a trimmed slice of GLUE
   **SST-2**, log to MLflow, register to Unity Catalog, tag the version `@champion`.
2. `deploy.py` — resolve `@champion` and serve it on the smallest CPU endpoint
   (`Small`, scale-to-zero).

`invoke_endpoint.py` then queries the endpoint locally.

> Refs: [AI Runtime CLI](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/) ·
> [Model Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/)

## Files

| File | Purpose |
| --- | --- |
| `train.py` | Fine-tune on GPU, log + register to UC, set `@champion` alias. |
| `deploy.py` | Deploy the aliased version to a Small, scale-to-zero endpoint. |
| `invoke_endpoint.py` | Classify sample sentences against the endpoint (local). |
| `train_deploy.yaml` | `air` workload: GPU compute, deps, chained train→deploy. |

## Prerequisites

- `air` installed and a Databricks profile authenticated (see the
  [intro sample](../ai-runtime-cli-intro/)).
- A Unity Catalog `catalog.schema` you can write models to, and permission to
  create serving endpoints.
- `pip install databricks-sdk` for local invocation.

## Configure

Everything is parameterized via the exports in `train_deploy.yaml` — edit these,
no code changes needed:

```yaml
export MODEL_NAME=main.default.distilbert_sentiment   # a UC path you can write to
export ENDPOINT_NAME=air-transformers-sentiment
export MODEL_ALIAS=champion                           # train→deploy handoff
export WORKLOAD_SIZE=Small                            # smallest workload size
export SCALE_TO_ZERO=true
```

## Run

```bash
# train + deploy (one chained run)
COPYFILE_DISABLE=1 air run --file train_deploy.yaml --watch -p my-workspace

# query the endpoint once it's ready
python invoke_endpoint.py --endpoint air-transformers-sentiment -p my-workspace
```

Example output:

```
[positive]  score=0.9987  I loved this movie, it was fantastic!
[negative]  score=0.9975  The plot was boring and the acting was terrible.
```

Pass your own text with repeated `--text` flags.

## Notes

- The fine-tune is trimmed (`--train-size 2048`, 1 epoch) to finish in a couple
  of minutes; scripts expose all knobs as CLI flags with env-var defaults.
- The pipeline is moved to CPU before logging so the CPU endpoint loads it with
  no GPU dependency.
- `air` can report `SUCCESS` even when a serverless job couldn't write UC
  artifacts (network-restricted workspaces). Verify the model version is `READY`
  and the endpoint exists, not just the job status.
- `COPYFILE_DISABLE=1` (macOS) keeps `._*` files out of the snapshot tarball.
