# Databricks-Samples
Data & ML Engineering samples in the Databricks ecosystem.

This repository contains **hands-on code samples** and **conceptual guides** that accompany my YouTube videos, focused on Databricks, MLflow, Model Serving, and Foundation Models.

---

## 📘 Intro Concepts & Platform Overview

High-level explanations for folks new to Databricks, Cloud, and the modern data + ML stack.

- [What is Databricks](https://www.youtube.com/watch?v=5KRrw2qdtlg&t=354s)

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

## ⭐️ Feedback & Contributions

If you find these samples useful:
- Star the repo
- Open an issue for bugs or suggestions
- Feel free to fork and extend for your own projects
