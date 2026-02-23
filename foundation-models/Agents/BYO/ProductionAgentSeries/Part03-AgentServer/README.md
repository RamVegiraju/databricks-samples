# AgentServing Sample

Minimal MLflow Agent Server sample using:
- `@invoke` for non-stream responses
- `@stream` for SSE streaming responses
- A simple tool (`get_weather`) with Databricks model calls via LangChain

## What this sample covers

- Local Agent Server startup (`start_server.py`)
- Non-stream + stream request handling (`agent.py`)
- End-to-end client invocation examples (`requests.py`)
- Tool calling and tool output events in Responses API format

## Prerequisites

- Python environment with dependencies installed
- Databricks auth configured in shell:

```bash
export DATABRICKS_HOST="https://<your-workspace>"
export DATABRICKS_TOKEN="<your-token>"
```

Optional model config:

```bash
export AGENT_LLM_ENDPOINT="databricks-gpt-oss-120b"
export AGENT_LLM_TEMPERATURE="0.1"
```

## Start the server

```bash
cd mlflow/AgentServing
python3 start_server.py --reload --port 8000
```

## Invoke (non-stream)

```bash
curl -sS http://127.0.0.1:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input":[{"role":"user","content":"What is the weather in Seattle?"}]}' | jq
```

## Invoke (stream)

```bash
curl -N --no-buffer -sS http://127.0.0.1:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input":[{"role":"user","content":"What is the weather in Seattle?"}], "stream": true}'
```

## Python client (both modes)

```bash
python3 requests.py --base-url http://127.0.0.1:8000
```

## Credits/Additional Resources

- MLflow Agent Server docs: https://mlflow.org/docs/latest/genai/serving/agent-server/
- MLflow ResponsesAgent docs: https://mlflow.org/docs/latest/genai/serving/responses-agent/
