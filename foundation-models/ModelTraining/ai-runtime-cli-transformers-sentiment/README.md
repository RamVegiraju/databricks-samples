# AI Runtime CLI — Fine-tune a Transformers Model, then Deploy to Model Serving

Fine-tune DistilBERT for sentiment analysis on a serverless **GPU** with the
**AI Runtime CLI** (`air`), then deploy the trained model to a **CPU** Model
Serving endpoint — as **two decoupled steps**:

1. **Train (GPU)** — `air run --file train.yaml` runs `train.py` on a single A10:
   fine-tune `distilbert-base-uncased` on a trimmed slice of GLUE **SST-2**, log
   to MLflow, register to Unity Catalog, and tag the new version `@champion`.
2. **Deploy (local, no GPU)** — `python deploy.py` resolves `@champion` and
   serves it on the smallest CPU endpoint (`Small`, scale-to-zero), polling until
   the endpoint is READY.
3. **Invoke** — `python invoke_endpoint.py` queries the endpoint locally.

## Why two steps?

Training needs the A10; **deployment does not**. Creating a serving endpoint is a
control-plane operation (a few SDK calls) — so `deploy.py` runs on your laptop
against your Databricks CLI profile, and the GPU is released the moment training
finishes. Chaining deploy *inside* the GPU job (the older pattern) keeps the A10
allocated and billed for the entire ~10–20 min endpoint build, and an automatic
job retry would needlessly re-run training. Keeping the steps independent avoids
both.

> Refs: [AI Runtime CLI](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/) ·
> [Model Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/)

## Files

| File | Purpose |
| --- | --- |
| `train.py` | Fine-tune on GPU, log + register to UC, set `@champion` alias. |
| `train.yaml` | `air` workload: GPU compute, pinned deps, runs `train.py` only. |
| `deploy.py` | Resolve `@champion` → CPU endpoint; poll to READY (runs locally). |
| `invoke_endpoint.py` | Classify sample sentences against the endpoint (local). |

## Prerequisites

- `air` installed and a Databricks profile authenticated via OAuth
  (`databricks auth login --profile <profile>`); see the
  [intro sample](../ai-runtime-cli-intro/).
- A Unity Catalog `catalog.schema` you can write models to, and permission to
  create serving endpoints.
- `pip install databricks-sdk` locally (used by `deploy.py` and
  `invoke_endpoint.py`; both auth via your CLI profile).

## Configure — everything is parameterized

**Training** config lives in the `export`s in `train.yaml` (the single source of
truth `train.py` reads via its env-var defaults). Every `train.py` hyperparameter
is also a CLI flag (`--base-model`, `--train-size`, `--epochs`, `--lr`, …).

```yaml
export MODEL_NAME=custom_ml.models.distilbert_sentiment   # a UC path you can write to
export MODEL_ALIAS=champion                               # train→deploy handoff
```

**Deploy** config is CLI flags on `deploy.py`, each defaulting to an env var so
the two steps share one source of truth:

| Flag | Env var | Default |
| --- | --- | --- |
| `--model-name` | `MODEL_NAME` | `custom_ml.models.distilbert_sentiment` |
| `--alias` | `MODEL_ALIAS` | `champion` |
| `--version` | — | (resolve `--alias`) |
| `--endpoint` | `ENDPOINT_NAME` | `air-transformers-sentiment` |
| `--workload-size` | `WORKLOAD_SIZE` | `Small` |
| `--workload-type` | `WORKLOAD_TYPE` | `CPU` |
| `--scale-to-zero` / `--no-scale-to-zero` | `SCALE_TO_ZERO` | `true` |
| `--interval` | — | `90` (seconds between readiness polls) |
| `--timeout-min` | — | `40` |

## Run

```bash
# 1. Train + register on the GPU (finishes in a couple of minutes)
COPYFILE_DISABLE=1 air run --file train.yaml --watch -p my-workspace

# 2. Deploy the @champion version to a CPU endpoint (local; polls to READY)
python deploy.py -p my-workspace

# 3. Query the endpoint
python invoke_endpoint.py --endpoint air-transformers-sentiment -p my-workspace
```

Example output:

```
[positive]  score=0.8691  An absolute masterpiece, highly recommend.
[negative]  score=0.8375  Waste of time, I want my money back.
```

Pass your own text with repeated `--text` flags.

## Notes

- **CPU-deployable artifact.** `train.py` moves the pipeline to CPU
  (`model.cpu()` + `pipeline(..., device=-1)`) *before* logging, so the CPU
  serving endpoint loads it with no CUDA dependency. The saved weights are
  device-agnostic; Model Serving loads them on CPU at serve time.
- **Pinned Hub deps.** `train.yaml` pins `datasets==3.1.0`,
  `huggingface_hub==0.26.2`, `fsspec==2024.9.0`, and adds `hf_transfer`. The
  runtime's unpinned resolve otherwise installs a `datasets`/`huggingface_hub`
  pair that disagree on `HfFileSystem.find(maxdepth=...)` (breaks
  `load_dataset`), and it enables `HF_HUB_ENABLE_HF_TRANSFER=1` without shipping
  `hf_transfer`.
- **Namespaced dataset id.** The dataset defaults to `nyu-mll/glue` (config
  `sst2`); recent `datasets` releases reject the bare `glue` id.
- The fine-tune is trimmed (`--train-size 2048`, 1 epoch) to finish in ~2 min on
  a single A10.
- First endpoint build takes ~10–20 min (container build + compute); `deploy.py`
  prints endpoint state every `--interval` seconds while it waits. Re-running
  `deploy.py` after a new training run updates the endpoint in place.
- `COPYFILE_DISABLE=1` (macOS) keeps `._*` files out of the snapshot tarball.
