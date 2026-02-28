# Part 04 — Lakebase Managed Memory

A Streamlit chatbot backed by `databricks-gpt-oss-120b` with two-tier persistent
memory stored entirely in a Databricks Lakebase instance. No external vector
database. No LangGraph. Raw SQL and pgvector.

---

## Architecture

```
User logs in
  |
  +-- Short-term: load message history for this session (messages table)
  +-- Long-term:  embed first message, retrieve top-5 relevant past summaries (pgvector)
  |
  +--> Build context: [system prompt] + [long-term memories] + [short-term history]
  +--> Call databricks-gpt-oss-120b (Databricks AI Gateway)
  +--> Persist reply to messages table
  |
User logs out
  +-- Summarize session with databricks-gpt-oss-120b
  +-- Embed summary with bge-large-en (AI Gateway)
  +-- Store in long_term_memory table (feeds future sessions)
```

### Short-term memory

Every message turn is written to and read from the `messages` table on each
request. This gives the model full context of the current session. Scoped to
`session_id` — a new session starts clean.

### Long-term memory

On session end, the model summarizes the conversation. That summary is embedded
using `databricks-bge-large-en` (1024 dims) and stored in `long_term_memory`.

On the next login, the user's first message is embedded and a pgvector cosine
similarity search retrieves the top-5 most relevant past summaries. These are
injected as a system-level context block before the conversation begins.

```
New session start
  first message -> embed -> pgvector search -> top-5 summaries -> system context

Each turn
  user message  -> messages table (write)
  full history  -> messages table (read)  -> sent to model

Session end
  all messages  -> summarize -> embed -> long_term_memory (write)
```

---

## Schema

```
databricks_postgres  (default Lakebase database)
  |
  +-- users               one row per user
  |     user_id           UUID PK
  |     username          TEXT UNIQUE
  |     created_at
  |
  +-- sessions            one row per login
  |     session_id        UUID PK
  |     user_id           FK -> users
  |     started_at
  |     ended_at          NULL = session still active
  |
  +-- messages            short-term memory
  |     message_id        UUID PK
  |     session_id        FK -> sessions
  |     user_id           FK -> users (denormalized for fast per-user queries)
  |     role              'user' | 'assistant'
  |     content           TEXT
  |     token_count       completion tokens (optional)
  |     created_at
  |
  +-- long_term_memory    long-term memory
        user_id           FK -> users   (PK composite)
        session_id        FK -> sessions (PK composite)
        summary           TEXT  (model-generated, 3-5 sentences)
        embedding         vector(1024)  (bge-large-en, indexed via HNSW)
        created_at
```

**Why `user_id` on `messages`?**
Denormalized so you can query a user's full history across sessions without
joining through `sessions` every time.

**Why HNSW index on `embedding`?**
HNSW (Hierarchical Navigable Small World) is an approximate nearest-neighbour
index. Much faster than a sequential scan as the table grows. Configured for
cosine distance (`vector_cosine_ops`) which matches how bge embeddings are
compared.

---

## Infrastructure

| Component | What it is |
|---|---|
| Lakebase instance | Databricks managed PostgreSQL server (CU_1) |
| Database | `databricks_postgres` (default, no extra DB needed) |
| pgvector | PostgreSQL extension for vector storage + similarity search |
| Chat model | `databricks-gpt-oss-120b` via [Databricks AI Gateway](https://docs.databricks.com/aws/en/ai-gateway/) |
| Embeddings | `databricks-bge-large-en` via [Databricks AI Gateway](https://docs.databricks.com/aws/en/ai-gateway/) |
| Auth | Databricks OAuth short-lived tokens (regenerated at connect time) |

Both the chat model and embeddings use the same `DATABRICKS_BASE_URL` AI Gateway
endpoint — the OpenAI client routes by model name.

---

## Files

```
schema.sql            DDL for all four tables + indexes
db.py                 All database operations (psycopg2, raw SQL)
memory.py             AI operations: embed() and summarize() — no DB logic
app.py                Streamlit chatbot UI
provision_lakebase.py One-time script to create the Lakebase instance via SDK
apply_schema.py       One-time script to apply schema.sql to the database
workspace_connection.py  Validate Databricks SDK connectivity
requirements.txt
.env                  Credentials (not committed)
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure credentials

Create a `.env` file in this directory:

```bash
# Databricks workspace
DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com
DATABRICKS_TOKEN=<your-pat-token>

# AI Gateway base URL — find this in your workspace under Serving > AI Gateway
# Format: https://<workspace-id>.ai-gateway.cloud.databricks.com/mlflow/v1
DATABRICKS_BASE_URL=https://<workspace-id>.ai-gateway.cloud.databricks.com/mlflow/v1

# Embeddings endpoint name
DATABRICKS_EMBEDDING_ENDPOINT=databricks-bge-large-en

# Lakebase (PostgreSQL) — fill in after provisioning
LAKEBASE_INSTANCE_NAME=<your-instance-name>
LAKEBASE_HOST=<instance-id>.database.cloud.databricks.com
LAKEBASE_DATABASE=databricks_postgres
LAKEBASE_USER=<your-databricks-email>
```

> **Finding `DATABRICKS_BASE_URL`:** In your workspace go to **Serving → AI Gateway**,
> or use the URL format shown in the model serving playground when you select a
> foundation model endpoint.

### 3. Provision the Lakebase instance (first time only)

If you don't have a Lakebase instance yet:

```bash
python3 provision_lakebase.py
```

This creates a `chatbot-memory` instance and prints the `LAKEBASE_HOST` value
to add to your `.env`. If you already have an instance, skip this step and fill
in the `.env` values from the Databricks UI (**Compute → Lakebase**).

### 4. Apply the schema (first time only)

```bash
python3 apply_schema.py
```

This enables the `pgvector` extension and creates the four tables
(`users`, `sessions`, `messages`, `long_term_memory`) plus indexes.

### 5. Run

```bash
streamlit run app.py
```

Open http://localhost:8501, enter a username, and start chatting. Long-term
memory is written when you click **Logout** at the end of a session.

---

## Key design decisions

- **One database, no extra schemas.** Everything lives in `databricks_postgres`
  under the `public` schema. No multi-tenant isolation needed for this use case.

- **No LangGraph.** Long-term memory is two SQL methods: `save_long_term_memory`
  and `get_relevant_memories`. The pgvector `<=>` operator handles similarity
  ranking. No framework needed.

- **Long-term retrieved once per session.** The top-5 memories are fetched on
  the first user message and held for the session. Short-term is read/written on
  every turn.

- **Tokens are short-lived.** Lakebase OAuth tokens expire in ~60 minutes.
  `db.py` regenerates and reconnects automatically when the token is within
  5 minutes of expiry.

- **Reasoning model content blocks.** `databricks-gpt-oss-120b` returns
  structured content (reasoning + text blocks). The app extracts only the `text`
  blocks before display and storage.
