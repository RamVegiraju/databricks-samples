# ResponsesAgentInterface: Local Serve + Test

This folder contains a LangGraph agent wrapped in the MLflow `ResponsesAgent` interface (`agent.py`), plus helper scripts to log and test the model locally.

## 1) Prerequisites

- Python virtual environment with required dependencies installed
- Databricks workspace access
- Databricks token (PAT or supported auth)
- `DATABRICKS_HOST` for your workspace

## 2) Configure environment variables

Set these in your terminal before logging / serving:

```bash
export DATABRICKS_HOST="https://<your-workspace>"
export DATABRICKS_TOKEN="<your-token>"
export MLFLOW_TRACKING_URI="databricks"
export DBX_EXPERIMENT_PATH="/Users/<your-user>/ResponsesAgentInterface"
export MLFLOW_EXPERIMENT_NAME="/Users/<your-user>/ResponsesAgentInterface"
```

Notes:
- Do not hardcode secrets in source files.
- `DBX_EXPERIMENT_PATH` is used by `log_model.py` when creating/selecting the experiment.

## 3) Log the model to MLflow

From this directory:

```bash
python3 log_model.py
```

Expected output includes:
- `TRACKING_URI: databricks`
- `EXPERIMENT: ...`
- `RUN_ID: ...`
- `MODEL_URI: models:/m-...`

Copy the printed `MODEL_URI`.

## 4) Serve locally

```bash
mlflow models serve -m "<MODEL_URI>" -p 8000 --env-manager local
```

Example:

```bash
mlflow models serve -m "models:/m-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" -p 8000 --env-manager local
```

## 5) Test inference

### Option A: curl

```bash
curl -s http://127.0.0.1:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input":[{"role":"user","content":"What is the weather in Seattle?"}]}' | jq
```

### Option B: Python requests client

Use the provided `requests_client.py` script (see next section).

## 6) Run the sample requests client

`requests_client.py` includes:
- a non-tool example prompt
- a tool-triggering prompt (`get_weather`)

Run both:

```bash
python3 requests_client.py --base-url http://127.0.0.1:8000
```

Run one mode only:

```bash
python3 requests_client.py --mode no-tool
python3 requests_client.py --mode tool
```

## Troubleshooting

- **Model not found for `models:/m-...`**
  - Ensure `MLFLOW_TRACKING_URI=databricks` is set in the serve terminal.
- **401 auth errors**
  - Re-check `DATABRICKS_HOST` and token validity for that workspace.
- **Experiment not visible in Databricks**
  - Verify `DBX_EXPERIMENT_PATH` points to a valid `/Users/<user>/...` path.
- **Tracing warning about missing experiment id**
  - Set `MLFLOW_EXPERIMENT_NAME` (or `MLFLOW_EXPERIMENT_ID`) in the serve terminal.

## Credits/Additional Resources

- MLflow Agent Server docs: https://mlflow.org/docs/latest/genai/serving/agent-server/
- MLflow ResponsesAgent docs: https://mlflow.org/docs/latest/genai/serving/responses-agent/
