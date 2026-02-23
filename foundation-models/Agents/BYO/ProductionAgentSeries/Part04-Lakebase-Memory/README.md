# Lakebase Managed Memory

A terminal chatbot backed by Claude (via Databricks Foundation Model APIs) with
two-tier persistent memory stored entirely in a Databricks Lakebase instance.
No external vector database. No LangGraph. Raw SQL and pgvector.

---

## Architecture

```
User logs in
  |
  +-- Short-term: load message history for this session (messages table)
  +-- Long-term:  embed first message, retrieve top-5 relevant past summaries (pgvector)
  |
  +--> Build context: [system prompt] + [long-term memories] + [short-term history]
  +--> Call Claude (Databricks FM API)
  +--> Persist reply to messages table
  |
User logs out
  +-- Summarize session with Claude
  +-- Embed summary with bge-embed-ram
  +-- Store in long_term_memory table (feeds future sessions)
```

### Short-term memory

Every message turn is written to and read from the `messages` table on each
request. This gives Claude the full context of the current session. Scoped to
`session_id` — a new session starts clean.

### Long-term memory

On session end, Claude summarizes the conversation. That summary is embedded
using `databricks-bge-large-en` (1024 dims) and stored in `long_term_memory`.

On the next login, the user's first message is embedded and a pgvector cosine
similarity search retrieves the top-5 most relevant past summaries. These are
injected as a system-level context block before the conversation begins.

```
New session start
  first message -> embed -> pgvector search -> top-5 summaries -> system context

Each turn
  user message  -> messages table (write)
  full history  -> messages table (read)  -> sent to Claude

Session end
  all messages  -> Claude summarize -> embed -> long_term_memory (write)
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
        summary           TEXT  (Claude-generated, 3-5 sentences)
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
| Claude | `databricks-claude-opus-4-6` via Databricks FM API (AI Gateway) |
| Embeddings | `databricks-bge-large-en` via Databricks Model Serving endpoint |
| Auth | Databricks OAuth short-lived tokens (regenerated at connect time) |

---

## Files

```
schema.sql            DDL for all four tables + indexes
db.py                 All database operations (psycopg2, raw SQL)
memory.py             AI operations: embed() and summarize() — no DB logic
test_chat.py          Terminal chat loop wiring db.py + memory.py + Claude
provision_lakebase.py One-time script to create the Lakebase instance via SDK
workspace_connection.py  Validate Databricks SDK connectivity
requirements.txt
.env                  Credentials (not committed)
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env
# Fill in DATABRICKS_HOST, DATABRICKS_TOKEN,
# DATABRICKS_BASE_URL, DATABRICKS_EMBEDDING_ENDPOINT,
# LAKEBASE_HOST, LAKEBASE_DATABASE, LAKEBASE_USER, LAKEBASE_INSTANCE_NAME

# 3. Provision Lakebase instance (first time only)
python3 provision_lakebase.py

# 4. Apply schema
# Connect to databricks_postgres and run schema.sql
# (provision_lakebase.py does this automatically)

# 5. Run
python3 test_chat.py
```

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
