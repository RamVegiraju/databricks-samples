# Part 06 — Full-Stack Production Agent

This part brings together every concept from the series into a single deployable system: a LangGraph agent served via AgentServer and the ResponsesAgent interface, connected to a remote MCP server, backed by Lakebase long-term memory and checkpointed short-term memory, with MLflow GenAI evaluation wired in.

```
Part06-Full-Stack/
├── provision_lakebase.py        # Run once before deploying — creates Lakebase instance
├── mcp_server/                  # Standalone Databricks App — deploy first
│   ├── app.yaml
│   ├── pyproject.toml
│   └── server/
│       ├── app.py               # FastMCP + FastAPI, OBO auth middleware
│       ├── tools.py             # MCP tools: add, return_biodata
│       ├── utils.py             # Databricks App header / OBO auth helpers
│       └── main.py              # uvicorn entry point
│
├── agent_app/                   # Main Databricks App — deploy second
│   ├── app.yaml
│   ├── pyproject.toml
│   ├── .env.example
│   ├── start_server.py          # AgentServer startup
│   └── agent/
│       ├── tools.py             # Inline Python tool: get_product_info
│       ├── memory.py            # DatabricksStore + CheckpointSaver + memory tools
│       ├── agent.py             # LangGraph create_react_agent, lazy async init
│       └── server.py            # @invoke / @stream, ResponsesAgent interface
│
└── eval/
    └── evaluate.py              # mlflow.genai.evaluate() with built-in judges
```

---

## How the parts connect

| Concept | Where it came from | Where it lives in Part06 |
|---|---|---|
| Custom MCP server | Part01 | `mcp_server/` — deployed as its own app |
| ResponsesAgent interface | Part02 | `agent/server.py` — `@invoke` / `@stream` |
| AgentServer | Part03 | `start_server.py` + `agent/server.py` |
| Lakebase long-term memory | Part04 | `agent/memory.py` — `DatabricksStore` |
| Session checkpointing | Part04 (short-term) | `agent/memory.py` — `CheckpointSaver` |
| MLflow tracing + evaluation | Part05 | `eval/evaluate.py` |

### Memory: Part04 custom vs. Part06 built-in

Part04 built a custom `db.py` (~400 lines) with raw psycopg2, manual OAuth token refresh, explicit schema DDL, and pgvector queries. Part06 replaces all of that with two SDK-managed abstractions from `databricks_langchain`:

- **`DatabricksStore`** — long-term, cross-session memory with built-in semantic search. Implements LangGraph's `BaseStore` interface. Passed as `store=` to the agent.
- **`CheckpointSaver`** — short-term, within-session memory. Checkpoints full graph state to Postgres after every node. Passed as `checkpointer=` to the agent.

Both handle connection pooling and token refresh automatically. No manual schema DDL is required — `store.setup()` and `checkpointer.setup()` create all required tables at agent startup. Part04 is still the best reference for understanding what is happening under the hood.

---

## Prerequisites

- Databricks workspace with Foundation Model API access
- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/install.html) v0.200+ installed
- Python 3.11+

No prior deployments from other parts are needed — everything is provisioned and deployed from this directory.

---

## End-to-end setup

Complete these steps in order. Each step unlocks values needed by the next.

### 1 — Authenticate the CLI

```bash
databricks configure --host https://<your-workspace>.databricksapps.com
# or OAuth:
databricks auth login --host https://<your-workspace>.databricksapps.com
```

Verify: `databricks current-user me`

### 2 — Create your .env file

```bash
cp agent_app/.env.example agent_app/.env
```

Open `agent_app/.env` and fill in `DATABRICKS_HOST` and `DATABRICKS_TOKEN` now. All other values are filled in as you complete the steps below.

| Variable | Description | Filled in |
|---|---|---|
| `DATABRICKS_HOST` | Workspace URL (e.g. `https://<your-workspace>.databricks.com`) | Now |
| `DATABRICKS_TOKEN` | Personal access token | Now |
| `LAKEBASE_INSTANCE_NAME` | Output from Step 3 | After Step 3 |
| `MCP_SERVER_URL` | Deployed MCP app URL + `/mcp` | After Step 5 |
| `AGENT_LLM_ENDPOINT` | FM API model name (default: `databricks-claude-sonnet-4-6`) | Before Step 6 |
| `AGENT_LLM_TEMPERATURE` | LLM temperature (default: `0.1`) | Before Step 6 |
| `DATABRICKS_EMBEDDING_ENDPOINT` | Embedding model (default: `databricks-bge-large-en`) | Before Step 6 |
| `EMBEDDING_DIMS` | Embedding dimensions (default: `1024`) | Before Step 6 |
| `MLFLOW_TRACKING_URI` | MLflow tracking URI (default: `databricks`) | Before Step 8 |
| `MLFLOW_EXPERIMENT_NAME` | MLflow experiment path (e.g. `/Users/<user>/Part06`) | Before Step 8 |

The FM API base URL is derived from `DATABRICKS_HOST` in code — no separate variable needed.

### 3 — Provision Lakebase

Lakebase provides the managed PostgreSQL instance for both long-term memory (`DatabricksStore`) and session checkpointing (`CheckpointSaver`). Run once from the Part06 root:

```bash
pip install databricks-sdk python-dotenv psycopg2-binary
python provision_lakebase.py

# Optional: customize name, compute units, and retention window
python provision_lakebase.py --name part06-agent-memory --capacity CU_2 --retention 14
```

Copy the printed `LAKEBASE_INSTANCE_NAME` value into `agent_app/.env`.

Table creation for both `DatabricksStore` and `CheckpointSaver` happens automatically at agent startup via `store.setup()` and `checkpointer.setup()` — no manual DDL needed.

### 4 — Test the MCP server locally (optional but recommended)

Before deploying, verify the MCP server starts and tools respond correctly.

In one terminal, start the server:

```bash
cd mcp_server/
pip install -e .
python server/main.py --port 8001
```

Verify it is up:

```bash
curl http://localhost:8001/healthz
# expected: {"ok": true}
```

In a second terminal, run the test client:

```bash
cd mcp_server/
python test_local.py
```

Expected output:

```
=== Available tools ===
[add, return_biodata]

=== add(7, 8) ===
15

=== return_biodata('Alice') ===
{'age': 30, 'gender': 'Female', 'occupation': 'Software Engineer', 'interests': ['Python', 'Databricks', 'Hiking']}
```

Once both pass, stop the local server (`Ctrl+C`) and proceed to deploy.

### 5 — Create and deploy the MCP server app

```bash
# From anywhere — just registers the app name with Databricks, no local files needed
databricks apps create part06-mcp-server

# From mcp_server/ — install deps and deploy
cd mcp_server/
pip install -e .
databricks apps deploy part06-mcp-server --source-code-path .
```

Once deployed, get the app URL from the Databricks Apps UI (Compute → Apps → part06-mcp-server) and add it to `agent_app/.env`:

```
MCP_SERVER_URL=https://part06-mcp-server-<workspace-id>.databricksapps.com/mcp
```

Verify the MCP server is running: `curl https://<mcp-app-url>/healthz`

### 6 — Configure and create the agent app

All environment variables are defined in `agent_app/app.yaml` and take effect on the first deployment — no UI or CLI `update` commands needed.

Before creating the app, fill in the two placeholder values in `agent_app/app.yaml`:

```yaml
- name: MCP_SERVER_URL
  value: "https://<your-mcp-app>.databricksapps.com/mcp"   # from Step 5

- name: MLFLOW_EXPERIMENT_NAME
  value: "/Users/<your-username>/Part06-Full-Stack"
```

`LAKEBASE_INSTANCE_NAME` uses `valueFrom: "database"` — it is injected at runtime from the Lakebase resource bound to the app. Add the resource before deploying:

1. Go to Databricks Apps UI → **Create app** → name it `part06-agent-app`
2. Under **Resources**, add a Lakebase resource → select `part06-agent-memory` → set the resource key to `database`

All other env vars (`MLFLOW_TRACKING_URI`, `AGENT_LLM_ENDPOINT`, `AGENT_LLM_TEMPERATURE`, `DATABRICKS_EMBEDDING_ENDPOINT`, `EMBEDDING_DIMS`) are already set as static values in `app.yaml` and require no changes.

### 7 — Deploy the agent app

```bash
# From anywhere — just registers the app name with Databricks, no local files needed
databricks apps create part06-agent-app

# From agent_app/ — install deps and deploy
cd agent_app/
pip install -e .
databricks apps deploy part06-agent-app --source-code-path .
```

The app starts on port 8000. Once running, get the agent app URL from the UI — this is your `/invocations` endpoint.

### 8 — Run evaluation

With `MLFLOW_EXPERIMENT_NAME` set in `agent_app/.env`:

```bash
cd eval/
pip install mlflow openai python-dotenv
python evaluate.py
```

Results appear in the Databricks UI under **Experiments → Part06-Full-Stack → Evaluation**.

---

## Local development

### Run the MCP server locally

```bash
cd mcp_server/
pip install -e .
python server/main.py --port 8001
```

Verify: `curl http://localhost:8001/healthz`

### Run the agent app locally

```bash
cd agent_app/
pip install -e .
cp .env.example .env   # fill in all values
python start_server.py --reload --port 8000
```

For local testing, set `MCP_SERVER_URL=http://localhost:8001/mcp` to point at the locally running MCP server.

### Invoke (non-streaming)

```bash
curl -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "input": [{"role": "user", "content": "What is MLflow?"}],
    "custom_inputs": {"thread_id": "session-1", "user_id": "alice"}
  }'
```

### Invoke (streaming)

```bash
curl -N --no-buffer -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "input": [{"role": "user", "content": "Add 7 and 8."}],
    "stream": true,
    "custom_inputs": {"thread_id": "session-1", "user_id": "alice"}
  }'
```

---

## Agent tool surface

| Tool | Type | Description |
|---|---|---|
| `add` | MCP (remote) | Integer addition — from `mcp_server/` |
| `return_biodata` | MCP (remote) | Person profile lookup — from `mcp_server/` |
| `get_product_info` | Python (inline) | Databricks product catalog — from `agent/tools.py` |
| `save_memory` | Memory (LangGraph store) | Persist a fact to long-term memory |
| `get_memories` | Memory (LangGraph store) | Semantic search over past memories |

### Memory scoping

```
thread_id  →  CheckpointSaver  →  short-term (full graph state, within session)
user_id    →  DatabricksStore  →  long-term  (namespace: ("memories", user_id))
```

Both are passed in `custom_inputs` on every request. The system prompt instructs the agent to call `get_memories` at the start of each conversation and `save_memory` when the user shares important context.

---

## Evaluation

```bash
cd eval/
# Requires agent_app/.env with DATABRICKS_HOST, DATABRICKS_TOKEN, and MLFLOW_EXPERIMENT_NAME
python evaluate.py
```

Evaluates `get_product_info` tool call accuracy using:

- **`ToolCallCorrectness`** (built-in LLM judge) — checks tool name and argument correctness via fuzzy matching
- **`response_is_informative`** (custom `@scorer`) — validates the final response is substantive

Results are logged to the MLflow experiment at `MLFLOW_EXPERIMENT_NAME` and viewable in the Databricks UI under **Experiments → your experiment → Evaluation**.

The evaluation runs standalone — it does not require either app server to be running. Tool definitions in `eval/evaluate.py` mirror `agent_app/agent/tools.py` so scores reflect production behavior directly.

---

## Series overview

| Part | Focus |
|---|---|
| Part01 | Custom MCP server with FastMCP + Databricks Apps |
| Part02 | ResponsesAgent interface — agnostic LangGraph wrapper |
| Part03 | AgentServer — `@invoke` / `@stream` with SSE streaming |
| Part04 | Lakebase long-term memory — custom PostgreSQL + pgvector |
| Part05 | MLflow tracing, metadata, and GenAI evaluation |
| **Part06** | **All of the above, deployed as a Databricks App** |
