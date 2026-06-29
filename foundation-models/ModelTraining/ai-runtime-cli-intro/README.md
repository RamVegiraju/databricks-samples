# AI Runtime CLI Intro — PyTorch GPU Example

A minimal, end-to-end example of training a PyTorch model on Databricks
serverless GPU compute with the **AI Runtime CLI** (`air`). The job runs on a
single A10 GPU and logs params, metrics, and the trained model to MLflow.

> Official reference: [Databricks AI Runtime CLI docs](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/).
> This sample mirrors that quickstart with a working GPU training script.

## Files

| File | Purpose |
| --- | --- |
| `train.py` | Trains a tiny MLP on synthetic data on the GPU; logs to MLflow. |
| `train.yaml` | `air` workload config: experiment, compute, environment, command. |
| `watch_run.py` | Polls a run to completion, then prints its Job + MLflow links. |

## 1. Install the CLI

`air` ships as the `databricks-air` package; install it with [`uv`](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # skip if uv is installed
uv tool install --force databricks-air --python 3.12
air --version
```

## 2. Authenticate (OAuth)

`air` reuses Databricks CLI profiles. Log in with OAuth (opens a browser) and
save it under a named profile:

```bash
databricks auth login \
  --host https://<your-workspace>.cloud.databricks.com \
  -p my-workspace

databricks auth profiles   # my-workspace should show Valid = YES
```

## 3. Submit the training job

From this directory:

```bash
COPYFILE_DISABLE=1 air run --file train.yaml --watch -p my-workspace
```

This uploads the directory snapshot, provisions a 1×A10 GPU, installs the
dependencies, and runs `train.py`. `--watch` streams logs until the run exits.

- `COPYFILE_DISABLE=1` (macOS only) stops the snapshot tarball from including
  AppleDouble (`._*`) metadata files, which otherwise break `$CODE_SOURCE_PATH`
  resolution on the GPU node.
- `air run` prints a **Job Run ID** — note it for the next step.

Everything is parameterized rather than hardcoded:

- **Workload** (`train.yaml`): `experiment_name`, `compute.accelerator_type` /
  `num_accelerators`, and `environment.dependencies`.
- **Hyperparameters** (`train.py`): all CLI flags (`--epochs`, `--lr`,
  `--batch-size`, `--hidden`, `--n-samples`, `--n-features`, `--seed`) with
  defaults. Override them in the `command:` line, e.g.
  `python $CODE_SOURCE_PATH/train.py --epochs 20 --lr 5e-4`.
- **Watcher** (`watch_run.py`): `-p/--profile`, `--experiment`, `--interval`,
  `--timeout` (no host/IDs baked in — links are derived from the run).

## 4. Watch a run and get result links

Instead of (or after) `--watch`, use the helper to block until completion and
print clickable links to the Job run page and the MLflow run:

```bash
python watch_run.py <job-run-id> -p my-workspace
```

Example output:

```
Result:   SUCCESS
Duration: 2m 11s
Job run:           https://<host>/?o=...#job/<job-id>/run/<run-id>
MLflow experiment: https://<host>/ml/experiments/<experiment-id>
MLflow run:        https://<host>/ml/experiments/<experiment-id>/runs/<mlflow-run-id>
```

Other useful commands:

```bash
air list runs --active -p my-workspace   # active runs
air get run <run-id>  -p my-workspace    # status
air logs    <run-id>  -p my-workspace    # logs
air cancel  <run-id>  -p my-workspace    # cancel
```

## 5. Interpret the results

**In the logs / console** you'll see the GPU and the training curve, e.g.:

```
Using device: cuda
GPU: NVIDIA A10G
epoch  1/10  loss=0.5480  acc=0.8024
...
epoch 10/10  loss=0.0137  acc=0.9995
```

Loss should fall and accuracy should climb toward ~1.0 — the synthetic task is
linearly separable, so a healthy run converges quickly. (`acc` here is **training**
accuracy on synthetic data; it measures that the optimization loop works, not
generalization.)

**On the MLflow run page** (link from `watch_run.py`):

- **Parameters** — the hyperparameters (`epochs`, `lr`, `batch_size`, `device`, …).
- **Metrics** — `train_loss` and `train_accuracy` logged per epoch; click either
  to see the curve over `step` (epoch).
- **Artifacts › model** — the logged PyTorch model (pickle format), loadable via
  `mlflow.pytorch.load_model(<model-uri>)`.

**On the Job run page** — `air` submits each workload as a **one-time Databricks
Job run** (`SUBMIT_RUN`), so it appears under **Job Runs**, not as a standing job
in the Workflows list. Use it to inspect compute, duration, and raw driver logs.

## Notes

- A run that prints `Training complete` but stays `RUNNING` usually means the
  process hung after training. This sample logs the model with
  `serialization_format="pickle"` and calls `mlflow.end_run()` to exit cleanly;
  avoid `input_example` here, since signature/serving validation can stall.
- Change compute in `train.yaml` via `compute.accelerator_type` /
  `num_accelerators`; add Python deps under `environment.dependencies`.
