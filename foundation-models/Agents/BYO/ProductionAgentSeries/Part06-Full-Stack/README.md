# Part 06 — Full-Stack Production Agent

This part brings together every concept from the series into a single deployable system: a LangGraph agent served via AgentServer and the ResponsesAgent interface, connected to a remote MCP server, backed by Lakebase long-term memory and short-term session checkpointing, with MLflow GenAI evaluation wired in.

```
Part06-Full-Stack/
├── provision_lakebase.py         # Run once — creates Lakebase instance
├── setup_lakebase_permissions.py # Run once after app creation — grants SP permissions
├── grant_mcp_permissions.py      # Run once — grants agent app SP access to MCP server app
├── test_agent_app.py             # End-to-end test against deployed agent app
├── mcp_server/                   # Standalone Databricks App — deploy first
│   ├── app.yaml
│   ├── pyproject.toml
│   └── server/
│       ├── app.py                # FastMCP + FastAPI, OBO auth middleware
│       ├── tools.py              # MCP tools: add, return_biodata
│       ├── utils.py              # Databricks App header / OBO auth helpers
│       └── main.py               # uvicorn entry point
│
├── agent_app/                    # Main Databricks App — deploy second
│   ├── app.yaml
│   ├── pyproject.toml
│   ├── .env.example
│   ├── start_server.py           # AgentServer startup
│   └── agent/
│       ├── tools.py              # Inline Python tool: get_product_info
│       ├── memory.py             # AsyncDatabricksStore + AsyncCheckpointSaver + memory tools
│       ├── agent.py              # LangGraph create_react_agent, per-request init
│       └── server.py             # @invoke / @stream, ResponsesAgent interface
│
└── eval/
    └── evaluate.py               # mlflow.genai.evaluate() with built-in judges
```

---

## Architecture

```
[Client]
   │  POST /invocations  (ResponsesAgentRequest)
   ▼
[agent_app]  — Databricks App, port 8000
   │  start_server.py → AgentServer("ResponsesAgent") → FastAPI
   │  agent/server.py  → @invoke / @stream handlers
   │       │  Per-request: async with AsyncDatabricksStore as store
   │       │  Per-request: async with AsyncCheckpointSaver as checkpointer
   │  agent/agent.py   → init_agent(store, checkpointer) → create_react_agent
   │       ├── MCP tools (remote)      ← DatabricksMultiServerMCPClient
   │       ├── get_product_info        ← inline Python tool
   │       ├── get_user_memory         ← long-term memory (AsyncDatabricksStore)
   │       ├── save_user_memory        ← long-term memory (AsyncDatabricksStore)
   │       └── delete_user_memory      ← long-term memory (AsyncDatabricksStore)
   │  agent/memory.py  → AsyncDatabricksStore (long-term) + AsyncCheckpointSaver (short-term)
   │
   │  HTTP  (OBO auth via x-forwarded-access-token)
   ▼
[mcp_server] — Databricks App, port 8000
   │  server/app.py    → FastMCP + FastAPI
   │  server/tools.py  → add(a, b), return_biodata(name)
   └  server/utils.py  → OBO WorkspaceClient
```

---

## Components

### MCP Server (`mcp_server/`)

A standalone Databricks App that exposes tools over the Model Context Protocol (MCP). The agent app connects to it at runtime using `DatabricksMultiServerMCPClient`, which handles OAuth authentication automatically via the `x-forwarded-access-token` header injected by the Databricks Apps platform.

Tools:
- `add(a, b)` — integer addition
- `return_biodata(name)` — person profile lookup

### Agent App (`agent_app/`)

The main Databricks App. Exposes `/invocations` via MLflow AgentServer with two endpoints:
- `@invoke()` — non-streaming, returns a full `ResponsesAgentResponse`
- `@stream()` — streaming, yields `ResponsesAgentStreamEvent` via SSE

On every request, the handler opens `AsyncDatabricksStore` and `AsyncCheckpointSaver` as async context managers, builds the LangGraph agent, and streams the response. The agent is initialized per-request following the [app-templates](https://github.com/databricks/app-templates/tree/main/agent-langgraph-long-term-memory) pattern.

### Memory (`agent/memory.py`)

Two types of memory backed by a single Lakebase (managed PostgreSQL) instance:

| Type | Class | Scope | How |
|------|-------|-------|-----|
| Long-term | `AsyncDatabricksStore` | Cross-session, per `user_id` | pgvector semantic search via `DATABRICKS_EMBEDDING_ENDPOINT` |
| Short-term | `AsyncCheckpointSaver` | Within-session, per `thread_id` | Full LangGraph graph state checkpointed after every node |

Both are used as async context managers per-request. `store.setup()` and `checkpointer.setup()` are called on each request (idempotent — creates tables on first call only).

Memory tools access the store via `RunnableConfig["configurable"]["store"]` — no `InjectedStore` needed. The streaming handler populates config with `store`, `user_id`, and `thread_id` on every request.

### Memory tools

| Tool | Description |
|------|-------------|
| `get_user_memory(query)` | Semantic search over stored memories (top 5 by embedding similarity) |
| `save_user_memory(key, json)` | Persist a fact as a JSON object under the given key |
| `delete_user_memory(key)` | Remove a stored memory by key |

The system prompt instructs the agent to call `get_user_memory` at the start of every conversation and `save_user_memory` when the user shares important information.

---

## How the parts connect

| Concept | Where it came from | Where it lives in Part06 |
|---|---|---|
| Custom MCP server | Part01 | `mcp_server/` — deployed as its own app |
| ResponsesAgent interface | Part02 | `agent/server.py` — `@invoke` / `@stream` |
| AgentServer | Part03 | `start_server.py` + `agent/server.py` |
| Lakebase long-term memory | Part04 | `agent/memory.py` — `AsyncDatabricksStore` |
| Session checkpointing | Part04 (short-term) | `agent/memory.py` — `AsyncCheckpointSaver` |
| MLflow tracing + evaluation | Part05 | `eval/evaluate.py` |

**Memory vs Part04:** Part04 built a custom `db.py` (~400 lines) with raw psycopg2, manual token refresh, and explicit DDL. Part06 replaces all of that with `AsyncDatabricksStore` and `AsyncCheckpointSaver` from `databricks_langchain` — both handle connection pooling and token refresh automatically, and create their tables via `setup()` on first run.

---

## Prerequisites

- Databricks workspace with Foundation Model API access
- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/install.html) v0.200+ installed
- Python 3.11+

No prior deployments from other parts are needed.

---

## End-to-end setup

Complete these steps in order.

### 1 — Authenticate the CLI

```bash
databricks auth login --host https://<your-workspace>.cloud.databricks.com
```

Verify: `databricks current-user me`

### 2 — Create your .env file

```bash
cp agent_app/.env.example agent_app/.env
```

Fill in `DATABRICKS_HOST` and `DATABRICKS_TOKEN`. Other values are filled in as you complete the steps below.

| Variable | Description | When |
|---|---|---|
| `DATABRICKS_HOST` | Workspace URL | Now |
| `DATABRICKS_TOKEN` | Personal access token | Now |
| `LAKEBASE_INSTANCE_NAME` | Output from Step 3 | After Step 3 |
| `MCP_SERVER_URL` | Deployed MCP app URL + `/mcp` | After Step 5 |
| `AGENT_LLM_ENDPOINT` | FM API model name (e.g. `databricks-gpt-oss-120b`) | Before Step 7 |
| `DATABRICKS_EMBEDDING_ENDPOINT` | Embedding model (default: `databricks-bge-large-en`) | Before Step 7 |
| `EMBEDDING_DIMS` | Embedding dimensions (default: `1024`) | Before Step 7 |

### 3 — Provision Lakebase

Lakebase provides the managed PostgreSQL instance for both memory types.

```bash
pip install databricks-sdk python-dotenv
python provision_lakebase.py

# Optional: customize name and capacity
python provision_lakebase.py --name part06-agent-memory --capacity CU_2
```

Copy the printed instance name into `agent_app/.env` as `LAKEBASE_INSTANCE_NAME`.

### 4 — Test the MCP server locally (optional)

```bash
cd mcp_server/
pip install -e .
python server/main.py --port 8001
# In a second terminal:
python test_local.py
```

Expected: tools `[add, return_biodata]` respond correctly.

### 5 — Deploy the MCP server app

```bash
databricks apps create part06-mcp-server

cd mcp_server/
pip install -e .
databricks sync . /Workspace/Users/<your-username>/part06-mcp-server
databricks apps deploy part06-mcp-server --source-code-path /Workspace/Users/<your-username>/part06-mcp-server
```

Get the app URL from **Compute → Apps → part06-mcp-server** and set in `agent_app/.env`:

```
MCP_SERVER_URL=https://<mcp-app-url>/mcp
```

Verify: `curl https://<mcp-app-url>/healthz` → `{"ok": true}`

### 6 — Configure `agent_app/app.yaml`

Set these two values before deploying:

| Variable | Value |
|---|---|
| `MCP_SERVER_URL` | MCP app URL from Step 5 with `/mcp` appended |
| `AGENT_LLM_ENDPOINT` | A model available in your workspace (e.g. `databricks-gpt-oss-120b`) |

> Check available models: `databricks serving-endpoints list`

### 7 — Create the agent app and bind Lakebase

The resource binding grants the app's service principal `CONNECT` access to Lakebase.

1. Go to **Compute → Apps** in the Databricks UI
2. Click **Create app** → name it `part06-agent-app`
3. Under **App resources** → click **+ Add resource**
4. Select your Lakebase instance (`part06-agent-memory`), permission `Can connect`, key `database`
5. Click **Create app**

### 8 — Grant Lakebase schema and table permissions

The resource binding only grants `CONNECT`. The app also needs `USAGE + CREATE` on the schema and DML on all tables. Run once after the app is created:

```bash
pip install databricks-langchain databricks-sdk python-dotenv
python setup_lakebase_permissions.py
```

This script:
1. Looks up the app's service principal UUID
2. Creates its Postgres role via `LakebaseClient`
3. Grants `USAGE, CREATE ON SCHEMA public` + `SELECT/INSERT/UPDATE/DELETE` on all 8 tables

> Tables created: `store`, `store_vectors`, `store_migrations`, `vector_migrations`, `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`

### 9 — Grant agent app access to the MCP server app

By default the agent app's service principal cannot call the MCP server app. Grant `CAN_USE`:

```bash
python grant_mcp_permissions.py
```

### 10 — Deploy the agent app

```bash
cd agent_app/
pip install -e .
databricks sync . /Workspace/Users/<your-username>/part06-agent-app
databricks apps deploy part06-agent-app --source-code-path /Workspace/Users/<your-username>/part06-agent-app
```

### 11 — Test end-to-end

```bash
# Ensure a valid OAuth session exists
databricks auth login --host https://<your-workspace>.cloud.databricks.com

python3 test_agent_app.py --app-url https://<agent-app-url>.databricksapps.com
```

This runs 5 tests covering every tool and both response modes:

| Test | Tool | Mode |
|---|---|---|
| `get_product_info` | inline Python tool | non-stream |
| `MCP add` | MCP remote tool | non-stream |
| `MCP biodata` | MCP remote tool | non-stream |
| `memory save → retrieve` | AsyncDatabricksStore | non-stream |
| `streaming` | inline Python tool | stream |

### 12 — Run evaluation

```bash
cd eval/
pip install mlflow openai python-dotenv
python evaluate.py
```

Set `MLFLOW_EXPERIMENT_NAME` in `agent_app/.env` first (e.g. `/Users/<user>/Part06`). Results appear in **Experiments → your experiment** in the Databricks UI.

---

## Local development

### Run MCP server locally

```bash
cd mcp_server/
pip install -e .
python server/main.py --port 8001
```

Verify: `curl http://localhost:8001/healthz`

### Run agent app locally

```bash
cd agent_app/
pip install -e .
cp .env.example .env   # fill in all values
python start_server.py --reload --port 8000
```

Set `MCP_SERVER_URL=http://localhost:8001/mcp` in `.env` for local MCP testing.

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
| `get_user_memory` | Memory (long-term) | Semantic search over stored memories |
| `save_user_memory` | Memory (long-term) | Persist a fact to long-term memory |
| `delete_user_memory` | Memory (long-term) | Remove a stored memory by key |

### Memory scoping

```
thread_id  →  AsyncCheckpointSaver  →  short-term (full graph state, within session)
user_id    →  AsyncDatabricksStore  →  long-term  (namespace: ("user_memories", user_id))
```

Both are passed in `custom_inputs` on every request.

---

## Evaluation

Evaluates `get_product_info` tool call accuracy using:

- **`ToolCallCorrectness`** (built-in LLM judge) — checks tool name and argument correctness
- **`response_is_informative`** (custom `@scorer`) — validates the final response is substantive

Runs standalone — neither app server needs to be running.

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
