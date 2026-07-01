# AI Runtime CLI — Train → Register → Serve → Invoke

End-to-end example that extends [`ai-runtime-cli-intro`](../ai-runtime-cli-intro):
train a PyTorch MLP on a serverless **A10 GPU** with the **AI Runtime CLI**
(`air`), **register** it to Unity Catalog, **deploy** it to a **Model Serving**
endpoint, and **invoke** the endpoint over REST. Every step is parameterized —
no hosts, versions, or names are hardcoded in the logic.

> Official reference: [Databricks AI Runtime CLI docs](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/)
> and [Model Serving docs](https://docs.databricks.com/machine-learning/model-serving/).

## Files

| File | Purpose |
| --- | --- |
| `train.py` | Trains the MLP on the GPU; logs **with a signature** and **registers to Unity Catalog** (both required for serving). |
| `train.yaml` | `air` workload config: experiment, GPU compute, deps, command. |
| `deploy.py` | Creates/updates a Model Serving endpoint from the registered UC model version and waits until `READY`. |
| `invoke.py` | Sends synthetic feature rows to the endpoint and prints logits / probabilities / classes. |

## Why this differs from the intro

The intro deliberately logs **without** a signature to avoid serving-validation
stalls. Serving needs the opposite, so `train.py` here:

1. builds a signature + input example from a small CPU batch (float32, `[-1, n_features]`),
2. sets the registry to `databricks-uc` and passes `registered_model_name`,
3. keeps `serialization_format="pickle"` (classic `torch.save`) to avoid the pt2 path.

## Prerequisites

```bash
uv tool install --force databricks-air --python 3.12   # the `air` CLI
databricks auth login --host https://<workspace>.cloud.databricks.com -p my-workspace
```

`deploy.py` / `invoke.py` use the Databricks SDK. Run them with any Python that
has it (e.g. `uv run --with databricks-sdk python deploy.py ...`).

## 1. Train + register (on the GPU)

The registered model name is the one value that must match across all three
steps. It lives in `train.yaml`'s `command:` and defaults to
`custom_ml.models.air_mlp_classifier`.

```bash
COPYFILE_DISABLE=1 air run --file train.yaml --watch -p my-workspace
```

`--watch` streams logs until the run exits. On success the log prints
`Registered <name> version <N>`. (`COPYFILE_DISABLE=1` is macOS-only — it stops
AppleDouble `._*` files from breaking `$CODE_SOURCE_PATH` on the node.)

Override hyperparameters in the `command:` line, e.g.
`python $CODE_SOURCE_PATH/train.py --registered-model-name custom_ml.models.air_mlp_classifier --epochs 20 --lr 5e-4`.

## 2. Deploy to a serving endpoint

```bash
uv run --with databricks-sdk python deploy.py -p my-workspace
```

Defaults: serves the **latest** version of `custom_ml.models.air_mlp_classifier`
to endpoint `air-mlp-classifier` on `CPU`/`Small` with scale-to-zero. All
overridable:

```bash
uv run --with databricks-sdk python deploy.py -p my-workspace \
  --model-name custom_ml.models.air_mlp_classifier \
  --version 2 --endpoint-name air-mlp-classifier \
  --workload-size Small --workload-type CPU --no-scale-to-zero
```

Creating an endpoint the first time builds a container — expect **~10-20 min**.
`deploy.py` polls until `READY` and then prints the invocation URL. Re-running
with a new `--version` updates the existing endpoint in place (a few minutes).

## 3. Invoke the endpoint

```bash
uv run --with databricks-sdk python invoke.py -p my-workspace
```

Sends synthetic `n_features`-wide rows and prints, per row, the raw **logit**,
the **sigmoid probability**, and the predicted **class** (0/1):

```
  row  logit      prob     class
  ---  ---------  -------  -----
    0    -2.1934   0.1004      0
    1     3.8720   0.9797      1
    ...
```

`--n-features` must match the model's input width (32 by default). Override
`--endpoint-name`, `--n-rows`, `--seed` as needed.

## Parameterization summary

| Knob | Where | Default |
| --- | --- | --- |
| Registered model name | `train.yaml` command / `deploy.py --model-name` / `invoke.py` width | `custom_ml.models.air_mlp_classifier` |
| Hyperparameters | `train.py` flags in `train.yaml` command | see `train.py` |
| GPU compute | `train.yaml` `compute.*` | `1 x GPU_1xA10` |
| Endpoint name | `deploy.py` / `invoke.py --endpoint-name` | `air-mlp-classifier` |
| Model version | `deploy.py --version` | latest registered |
| Workload | `deploy.py --workload-size/--workload-type/--(no-)scale-to-zero` | `Small` / `CPU` / scale-to-zero |
| Profile | `-p/--profile` on every command | `DEFAULT` |

## Troubleshooting

**Endpoint stuck in `NOT_READY` / pending forever.** The most common cause with
this model is a **GPU/CPU serialization mismatch**: if the model is pickled while
on the GPU (`.to("cuda")`), the CPU serving container can't deserialize the CUDA
tensors and the container crash-loops, so the endpoint never goes `READY`. This
sample avoids it by moving the model to CPU *before* `log_model` (see the
`model.to("cpu")` call in `train.py`). If you adapt the script, keep that step,
or serve on a `--workload-type GPU_SMALL` endpoint instead.

To inspect a stuck endpoint's build/service logs before deleting it:

```bash
databricks serving-endpoints get <name> -p my-workspace          # state + reason
# In the UI: Serving > <endpoint> > Logs (build logs + served-model logs)
```

## Cleanup

```bash
databricks serving-endpoints delete air-mlp-classifier -p my-workspace
```
