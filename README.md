# Databricks-Samples

Hands-on code samples for Databricks, MLflow, Model Serving, Foundation Models, and Agents. Companion repo for my [YouTube channel](https://www.youtube.com/@RamVegiraju).

---

## Repository Structure

```
├── foundation-models/
│   ├── fm-api-intro.ipynb                    # Foundation Model API intro
│   ├── RAG/                                  # Vector Search + LangChain RAG
│   └── Agents/
│       ├── AgentBricks/                      # Knowledge Assistant & Info Extraction
│       └── BYO/
│           ├── agents-model-serving/         # Agent deployment on Model Serving
│           ├── Custom-MCP-Server-Agent/      # MCP server + LangGraph agent
│           └── ProductionAgentSeries/        # 6-part production agent series
│
├── mlflow/
│   ├── Intro/                                # MLflow intro notebook
│   ├── AgentServing/                         # MLflow Agent Server scaffold
│   └── ResponsesAgentInterface/              # ResponsesAgent wrapper + local serve
│
├── traditional-ml/
│   └── ModelServing/
│       ├── Built-In-MLflow/                  # Sklearn & Transformers serving
│       ├── BYO/                              # Custom PyFunc model serving
│       └── Multi-Model-Serving/              # Multi-model endpoint patterns
│
└── apps/
    └── mcp/                                  # MCP server + client test scripts
```

---

## Foundation Models & Agents

### Foundation Model APIs
- [`foundation-models/fm-api-intro.ipynb`](foundation-models/fm-api-intro.ipynb) — Getting started with Databricks FM APIs

### RAG
- [`foundation-models/RAG/`](foundation-models/RAG/) — Vector Search setup + LangChain RAG pipeline

### AgentBricks
- [`AgentBricks/KnowledgeAssistant/`](foundation-models/Agents/AgentBricks/KnowledgeAssistant/) — Knowledge Assistant API sample
- [`AgentBricks/InformationExtraction/`](foundation-models/Agents/AgentBricks/InformationExtraction/) — Information Extraction agent

### BYO Agents
- [`agents-model-serving/`](foundation-models/Agents/BYO/agents-model-serving/) — Deploying agents on Databricks Model Serving
- [`Custom-MCP-Server-Agent/`](foundation-models/Agents/BYO/Custom-MCP-Server-Agent/) — Custom MCP server with LangGraph agent

### Production Agent Series

A 6-part series building a production-grade agent end-to-end: MCP tools, ResponsesAgent interface, Agent Server, Lakebase memory, MLflow evaluation, and full-stack deployment as a Databricks App.

[`foundation-models/Agents/BYO/ProductionAgentSeries/`](foundation-models/Agents/BYO/ProductionAgentSeries/)

| Part | Focus |
|------|-------|
| [Part01](foundation-models/Agents/BYO/ProductionAgentSeries/Part01-Custom-MCP-Servers/) | Custom MCP server with FastMCP + Databricks Apps |
| [Part02](foundation-models/Agents/BYO/ProductionAgentSeries/Part02-ResponsesAgentInterface/) | ResponsesAgent interface — framework-agnostic LangGraph wrapper |
| [Part03](foundation-models/Agents/BYO/ProductionAgentSeries/Part03-AgentServer/) | AgentServer with `@invoke` / `@stream` and SSE streaming |
| [Part04](foundation-models/Agents/BYO/ProductionAgentSeries/Part04-Lakebase-Memory/) | Lakebase memory — managed PostgreSQL + pgvector |
| [Part05](foundation-models/Agents/BYO/ProductionAgentSeries/Part05-Evaluation/) | MLflow tracing, metadata, and GenAI evaluation |
| [Part06](foundation-models/Agents/BYO/ProductionAgentSeries/Part06-Full-Stack/) | Full-stack deployment — all parts combined as a Databricks App |

---

## MLflow & MLOps

- [`mlflow/Intro/`](mlflow/Intro/) — MLflow intro notebook
- [`mlflow/ResponsesAgentInterface/`](mlflow/ResponsesAgentInterface/) — LangGraph agent wrapped with MLflow ResponsesAgent, local serve + request examples
- [`mlflow/AgentServing/`](mlflow/AgentServing/) — Minimal Agent Server scaffold

---

## Traditional ML — Model Serving

- [`Built-In-MLflow/dbx-serving-sklearn.ipynb`](traditional-ml/ModelServing/Built-In-MLflow/dbx-serving-sklearn.ipynb) — Sklearn model on Databricks Model Serving
- [`Built-In-MLflow/transformers-dbx-serving.ipynb`](traditional-ml/ModelServing/Built-In-MLflow/transformers-dbx-serving.ipynb) — Transformers model on Model Serving
- [`BYO/custom-model-pyfunc.ipynb`](traditional-ml/ModelServing/BYO/custom-model-pyfunc.ipynb) — Custom PyFunc model serving
- [`Multi-Model-Serving/multi-model-serving-intro.ipynb`](traditional-ml/ModelServing/Multi-Model-Serving/multi-model-serving-intro.ipynb) — Multi-model endpoint patterns

---

## Apps

- [`apps/mcp/`](apps/mcp/) — MCP server and client test scripts

---

## Video Links

| Topic | Video |
|-------|-------|
| What is Databricks | [Watch](https://www.youtube.com/watch?v=5KRrw2qdtlg&t=354s) |
| What is Unity Catalog | [Watch](https://www.youtube.com/watch?v=EBRxWCAvL7U) |
| MLflow Introduction | [Watch](https://www.youtube.com/watch?v=Uz4AcTKirPY) |
| Model Serving Theory | [Watch](https://www.youtube.com/watch?v=UmHISXgPhGk&t=2s) |
| Model Serving Hands-On (Sklearn) | [Watch](https://www.youtube.com/watch?v=V1S4PEzMW1s) |
| Transformers on Model Serving | [Watch](https://www.youtube.com/watch?v=mQUFMExtJXM) |
| Foundation Model API Intro | [Watch](https://www.youtube.com/watch?v=LOBHuX0EfaA) |
| RAG on Databricks (Theory) | [Watch](https://www.youtube.com/watch?v=cAWxG8rAto0) |
| RAG on Databricks (Hands-On) | [Watch](https://www.youtube.com/watch?v=npBvZnpYdLw) |
| Agentic Options on Databricks | [Watch](https://www.youtube.com/watch?v=dgOB7Fksi5E) |

---

## Credits

- [databricks/app-templates](https://github.com/databricks/app-templates) — Agent and app templates
- [Databricks AI/ML Documentation](https://docs.databricks.com/aws/en/machine-learning/) — Foundation Models, MLflow, Model Serving
- [Databricks Lakebase Documentation](https://docs.databricks.com/aws/en/oltp/) — Managed PostgreSQL for agent memory

---

## Feedback & Contributions

Star the repo, open an issue, or fork and extend for your own projects.
