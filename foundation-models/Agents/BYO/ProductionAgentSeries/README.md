# Production Agent Series

A step-by-step series that builds toward a production-ready agent stack. Each part adds one layer.

---

## Part 1 — Custom MCP Servers

**Adds:** A custom MCP (Model Context Protocol) server with tools.

- Run your own MCP server (e.g. `uvicorn`) that exposes tools.
- Example tools: `add(a, b)`, `return_biodata(name)`.
- Use this server from clients or later parts that support MCP.

---

## Part 2 — ResponsesAgent Interface

**Adds:** An agent in the MLflow ResponsesAgent format, ready to log and serve.

- LangGraph agent (e.g. tool-calling with `get_weather`) wrapped as an MLflow `ResponsesAgent`.
- Log the model to MLflow and serve it locally with `mlflow models serve`.
- Test with curl or the included requests client.

---

## Part 3 — Agent Server

**Adds:** A full Agent Server with streaming and non-streaming endpoints.

- MLflow Agent Server with `@invoke` (non-stream) and `@stream` (SSE).
- Same kind of tool (e.g. `get_weather`) and Databricks model calls via LangChain.
- End-to-end client examples for both invoke and stream.

---

## Part 4 — Lakebase Memory

**Adds:** Persistent memory for a chatbot, no external vector DB.

- Terminal chatbot using Claude via Databricks Foundation Model APIs.
- Two-tier memory in a Databricks Lakebase (PostgreSQL + pgvector):
  - **Short-term:** per-session message history in a `messages` table.
  - **Long-term:** session summaries embedded and stored; top-k retrieved by similarity at session start.
- Schema, provisioning, and chat loop are included; no LangGraph in this part.

---

## Order

Parts are independent but designed to be read in order: **Part 1 → Part 2 → Part 3 → Part 4**. Each folder has its own README and setup steps.
