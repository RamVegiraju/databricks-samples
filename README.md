# Databricks-Samples
Data & ML Engineering samples in the Databricks ecosystem.

This repository contains **hands-on code samples** and **conceptual guides** that accompany my YouTube videos, focused on Databricks, MLflow, Model Serving, and Foundation Models.

---

## 📘 Intro Concepts & Platform Overview

High-level explanations for folks new to Databricks, Cloud, and the modern data + ML stack.

- [What is Databricks](https://www.youtube.com/watch?v=5KRrw2qdtlg&t=354s)
- [What is Unity Catalog](https://www.youtube.com/watch?v=EBRxWCAvL7U)

---

## 🧪 MLflow & MLOps

Covers experiment tracking, model registry, and operationalizing ML workloads using MLflow and Databricks Model Serving.

### Conceptual + Intro
- [MLflow Introduction & Hands-On](https://www.youtube.com/watch?v=Uz4AcTKirPY)
- [MLflow & Databricks Model Serving – Theoretical Intro](https://www.youtube.com/watch?v=UmHISXgPhGk&t=2s)

### Hands-On
- [Model Serving Hands-On (Scikit-learn Sample)](https://www.youtube.com/watch?v=V1S4PEzMW1s)
- [Deloying Transformers Models on Databricks Model Serving](https://www.youtube.com/watch?v=mQUFMExtJXM)

### Newly Added Samples (Code)
- [`mlflow/ResponsesAgentInterface/`](mlflow/ResponsesAgentInterface/) - LangGraph agent wrapped with MLflow `ResponsesAgent`, local serve flow, and request examples (tool and non-tool calls).
- [`mlflow/AgentServing/`](mlflow/AgentServing/) - Minimal Agent Server sample scaffold.
- [`mlflow/Intro/`](mlflow/Intro/) - Intro notebook samples.

---

## 🤖 Foundation Models & LLMs

Using Databricks Foundation Model APIs and Model Serving to work with LLMs and other generative models.

- [Foundation Model API Intro](https://www.youtube.com/watch?v=LOBHuX0EfaA)


### Retrieval Augmented Generation (RAG) Workflows
Series walking through setting up Vector Search, LangChain with Databricks Foundation Model APIs, and MLflow for RAG evaluation.

- [RAG on Databricks Part 1 Theoretical](https://www.youtube.com/watch?v=cAWxG8rAto0&list=PLThJtS7RDkOeQ9RqUPzNUG-qnW4LNe4h0&index=3)
- [RAG on Databricks Part 2 Hands-On](https://www.youtube.com/watch?v=npBvZnpYdLw&list=PLThJtS7RDkOeQ9RqUPzNUG-qnW4LNe4h0&index=2)

### Agents

- [Agentic Options on Databricks](https://www.youtube.com/watch?v=dgOB7Fksi5E)

**Production Agent Series** — A step-by-step series that builds a production-grade agent with managed memory, evals, and serving wired end-to-end. Uses **MLflow Agent Server** for serving, **Lakebase** for persistent memory (short- and long-term), and is designed to deploy as a **Databricks App**.
→ [`foundation-models/Agents/BYO/ProductionAgentSeries/`](foundation-models/Agents/BYO/ProductionAgentSeries/)

| Part | Focus |
|------|-------|
| Part01 | Custom MCP server with FastMCP + Databricks Apps |
| Part02 | ResponsesAgent interface — agnostic LangGraph wrapper |
| Part03 | AgentServer — `@invoke` / `@stream` with SSE streaming |
| Part04 | Lakebase long-term memory — custom PostgreSQL + pgvector |
| Part05 | MLflow tracing, metadata, and GenAI evaluation |
| Part06 | Full-stack deployment — all parts combined as a Databricks App |

### Claude Code Integration

This repo uses [Claude Code](https://claude.ai/claude-code) with skills sourced from two official Databricks repositories:

- [**databricks/app-templates**](https://github.com/databricks/app-templates) — agent development, memory, tooling, and deployment patterns
- [**databricks-solutions/ai-dev-kit**](https://github.com/databricks-solutions/ai-dev-kit) — broader Databricks platform skills covering MLflow, Spark, SQL, Unity Catalog, and more

Skills are stored in `.claude/skills/` and are automatically available in Claude Code sessions. Below is the full list:

**Agent Development**

| Skill | Purpose |
|-------|---------|
| `quickstart` | First-time setup and Databricks authentication |
| `run-locally` | Run and test agents locally |
| `modify-langgraph-agent` | Modify LangGraph agent code, tools, and configuration |
| `modify-openai-agent` | Modify OpenAI SDK agent code, tools, and configuration |
| `add-tools-langgraph` | Add MCP servers, UC functions, Genie, and vector search (LangGraph) |
| `add-tools-openai` | Add MCP servers, UC functions, Genie, and vector search (OpenAI SDK) |
| `discover-tools` | Discover available tools and resources in the workspace |
| `deploy` | Deploy apps via Databricks Asset Bundles |
| `migrate-from-model-serving` | Migrate ResponsesAgent from Model Serving to Databricks Apps |

**Memory**

| Skill | Purpose |
|-------|---------|
| `lakebase-setup` | Configure Lakebase for agent memory storage |
| `agent-langgraph-memory` | Add long-term + short-term memory to LangGraph agents |
| `agent-openai-memory` | Add long-term + short-term memory to OpenAI SDK agents |

**MLflow**

| Skill | Purpose |
|-------|---------|
| `databricks-mlflow-evaluation` | MLflow 3 GenAI eval, scorers, and datasets |
| `instrumenting-with-mlflow-tracing` | Add MLflow tracing to agents and LLM apps |
| `retrieving-mlflow-traces` | Get, filter, and search MLflow traces |
| `analyze-mlflow-trace` | Debug a single MLflow trace |
| `analyze-mlflow-chat-session` | Debug multi-turn chat sessions |
| `querying-mlflow-metrics` | Query token usage, latency, and quality metrics |
| `searching-mlflow-docs` | Fetch MLflow documentation |
| `mlflow-onboarding` | Get started with MLflow |
| `agent-evaluation` | Evaluate and optimize agent output quality |

**Databricks Platform**

| Skill | Purpose |
|-------|---------|
| `databricks-config` | Configure Databricks profile and auth |
| `databricks-docs` | General Databricks documentation reference |
| `databricks-python-sdk` | Python SDK, Databricks Connect, CLI, REST API |
| `databricks-asset-bundles` | DAB project creation and multi-environment deployments |
| `databricks-model-serving` | Deploy and query Model Serving endpoints |
| `databricks-jobs` | Create, run, and schedule Databricks Jobs |
| `databricks-unity-catalog` | System tables, volumes, and data lineage |
| `databricks-vector-search` | Vector search endpoints/indexes for RAG |
| `databricks-genie` | Genie Spaces and natural language SQL |
| `databricks-dbsql` | Databricks SQL, warehouses, and advanced SQL features |
| `databricks-iceberg` | Managed/external Iceberg tables, IRC, Snowflake interop |
| `databricks-agent-bricks` | Knowledge Assistants and Supervisor Agents |
| `databricks-app-python` | Python web apps (Streamlit, Dash, FastAPI, etc.) |
| `databricks-app-apx` | Full-stack apps with APX (FastAPI + React) |
| `databricks-aibi-dashboards` | AI/BI dashboard creation |
| `databricks-lakebase-provisioned` | Lakebase Provisioned (managed PostgreSQL) |
| `databricks-lakebase-autoscale` | Lakebase Autoscaling with scale-to-zero and branching |
| `databricks-metric-views` | Unity Catalog metric views and KPI definitions |
| `databricks-spark-declarative-pipelines` | DLT/SDP pipelines, streaming tables, and materialized views |
| `databricks-spark-structured-streaming` | Spark Structured Streaming production patterns |
| `spark-python-data-source` | Custom Spark data source connectors |
| `databricks-zerobus-ingest` | Near real-time gRPC ingestion into Delta tables |
| `databricks-synthetic-data-generation` | Generate synthetic data with Faker and Spark |
| `databricks-unstructured-pdf-generation` | Generate synthetic PDFs for RAG evaluation |

Skills provide up-to-date SDK patterns and best practices directly in the development context, reducing errors and keeping implementations aligned with official Databricks tooling.

---

## 📂 Repository Structure (WIP)

Each folder in this repo aligns with a video or concept and is designed to be:
- Minimal
- Copy-paste friendly
- Easy to extend for real projects

More samples will be added over time as new videos are released.

---

## 📺 YouTube Channel

All samples are explained step-by-step on YouTube:  
👉 [*Ram Vegiraju*](https://www.youtube.com/watch?v=5KRrw2qdtlg&list=PLThJtS7RDkOeQ9RqUPzNUG-qnW4LNe4h0&index=6)

---

## Credits & Additional References

This repo builds on and borrows from the following official Databricks resources:

- [**databricks/app-templates**](https://github.com/databricks/app-templates) — official agent and app templates that provided the foundation for skills, patterns, and deployment workflows used throughout this repo
- [**Databricks Lakebase Documentation**](https://docs.databricks.com/aws/en/oltp/) — reference for managed PostgreSQL (Lakebase Provisioned and Autoscale) used for agent memory
- [**Databricks AI/ML Documentation**](https://docs.databricks.com/aws/en/machine-learning/) — reference for Foundation Models, MLflow, Model Serving, and the broader AI/ML platform

---

## ⭐️ Feedback & Contributions

If you find these samples useful:
- Star the repo
- Open an issue for bugs or suggestions
- Feel free to fork and extend for your own projects
