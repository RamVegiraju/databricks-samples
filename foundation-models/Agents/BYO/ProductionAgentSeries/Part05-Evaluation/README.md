# Part 5 — Evaluation & Tracing with MLflow

Minimalistic samples showing how to trace LLM calls on Databricks and evaluate tool-calling agents using MLflow's GenAI evaluation suite.

## Files

| File | Description |
|------|-------------|
| `intro_tracing.py` | Baseline: single FM API call with MLflow autolog |
| `trace_with_metadata.py` | Attach `user_id` and `session_id` as dedicated trace columns |
| `eval_agent_with_tools.py` | Evaluate a tool-calling agent with built-in MLflow scorers |

## Setup

**1. Install dependencies**

```bash
python -m venv venv && source venv/bin/activate
pip install openai mlflow databricks-sdk
```

**Production deployments — use the lightweight tracing SDK instead**

For containerized services, serverless functions, or any environment where install size matters, swap the full `mlflow` package for `mlflow-tracing`:

```bash
pip install openai mlflow-tracing databricks-sdk
```

`mlflow-tracing` is ~5 MB with 5–8 dependencies vs ~1000 MB for the full `mlflow` package. It exposes the same tracing APIs used in these samples (`@mlflow.trace`, `mlflow.update_current_trace`, `mlflow.openai.autolog`, `mlflow.search_traces`, `set_tracking_uri`, `set_experiment`) but drops non-tracing components (run management, model logging, model registry, MLflow Projects).

> Do not install both `mlflow` and `mlflow-tracing` in the same environment — they share the `mlflow` namespace and will cause version conflicts.

**2. Configure environment**

A `.env` file is provided with values pre-filled for this workspace. Source it before running any sample:

```bash
source <(grep -v '^#' .env | grep '=' | xargs -I {} echo export \"{}\")
```

Or export individually:

```bash
export DATABRICKS_HOST=...
export DATABRICKS_TOKEN=...
export DATABRICKS_BASE_URL=...       # AI Gateway: https://<workspace-host>/mlflow/v1
export MLFLOW_TRACKING_URI=databricks
export MLFLOW_EXPERIMENT_NAME=...
```

> `.env` is gitignored — credentials will not be committed.

---

## Samples

### 1. Baseline tracing — `intro_tracing.py`

Sends a single chat message through the Databricks FM API. With `mlflow.openai.autolog()` enabled, MLflow automatically captures the request/response as a trace in your Databricks experiment.

```bash
python intro_tracing.py
```

---

### 2. Traces with user/session metadata — `trace_with_metadata.py`

**What it demonstrates:** How to attach `user_id` and `session_id` to each MLflow trace so they appear as dedicated **User** and **Session** columns in the Databricks Traces UI and are filterable via `mlflow.search_traces()`.

**Key API:**
```python
@mlflow.trace
def chat(user_message, user_id, session_id):
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.user": user_id,
            "mlflow.trace.session": session_id,
        }
    )
    ...
```

**Why `mlflow.openai.autolog()` is intentionally absent here:**
`mlflow.openai.autolog()` creates its own top-level trace for every OpenAI call. When both autolog and `@mlflow.trace` are active, each invocation produces two separate traces — the autolog one does not inherit the `metadata` update, so `user`/`session` columns are missing from it. By letting `@mlflow.trace` own the trace lifecycle exclusively, every trace is guaranteed to carry the correct metadata.

**Querying by user or session:**
```python
mlflow.search_traces(filter_string="metadata.`mlflow.trace.user` = 'user-1'")
mlflow.search_traces(filter_string="metadata.`mlflow.trace.session` = 'session-aaa'")
```

**Run:**
```bash
python trace_with_metadata.py
```

---

### 3. Evaluate an agent with tools — `eval_agent_with_tools.py`

**What it demonstrates:** How to evaluate a tool-calling agent end-to-end using `mlflow.genai.evaluate()`. The agent calls a `get_weather` stub. Two scorers are applied: the built-in `ToolCallCorrectness` LLM judge and a custom `@scorer`.

**Why `@mlflow.trace(span_type=SpanType.TOOL)` is required:**
`ToolCallCorrectness` extracts tool calls by calling `trace.search_spans(span_type=SpanType.TOOL)`. `mlflow.openai.autolog()` creates `LLM`-typed spans for model calls but does **not** create `TOOL`-typed spans for local Python function executions. Without the explicit `SpanType.TOOL` annotation on the tool function, `actual_calls` is always empty and every row scores as incorrect.

```python
from mlflow.entities import SpanType

@mlflow.trace(span_type=SpanType.TOOL)
def get_weather(city: str) -> str:
    ...
```

**Dataset format** — `expectations.expected_tool_calls` drives fuzzy matching:
```python
{
    "inputs": {"query": "What's the weather in San Francisco?"},
    "expectations": {
        "expected_tool_calls": [
            {"name": "get_weather", "arguments": {"city": "San Francisco"}},
        ]
    },
}
```

> Note: if any expected call omits `"arguments"`, argument checking is skipped for **all** calls in that row (MLflow source behaviour). Include `"arguments"` on every entry to get per-argument scoring.

**Scorers:**
- `ToolCallCorrectness()` — LLM judge (fuzzy match by default; pass `should_exact_match=True` for strict comparison).
- `is_not_empty` — custom `@scorer` demo returning a `Feedback` object.

```python
from mlflow.genai.scorers import ToolCallCorrectness, scorer
from mlflow.entities import Feedback

@scorer
def is_not_empty(outputs: str) -> Feedback:
    ok = len(outputs.strip()) > 10
    return Feedback(value=ok, rationale="Response has content." if ok else "...")

results = mlflow.genai.evaluate(
    predict_fn=predict,
    data=eval_data,
    scorers=[ToolCallCorrectness(), is_not_empty],
)
print(results.metrics)
```

**Run:**
```bash
python eval_agent_with_tools.py
```

Results (per-row scores + aggregate metrics) are logged to your MLflow experiment and printed to stdout.
