-- Run once against databricks_postgres on the Lakebase instance.
-- Requires pgvector: CREATE EXTENSION IF NOT EXISTS vector;

-- ── Users ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    user_id    UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    username   TEXT        NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Sessions (one per login) ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    session_id UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID        NOT NULL REFERENCES users(user_id)    ON DELETE CASCADE,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at   TIMESTAMPTZ          -- NULL = session still active
);

-- ── Messages (short-term memory) ──────────────────────────────────────────
-- Every turn in the current session. Loaded on each request to build
-- the context window sent to Claude.
CREATE TABLE IF NOT EXISTS messages (
    message_id  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID        NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_id     UUID        NOT NULL REFERENCES users(user_id)       ON DELETE CASCADE,
    role        TEXT        NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT        NOT NULL,
    token_count INTEGER,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Long-term memory ──────────────────────────────────────────────────────
-- One row per completed session. Stores a Claude-generated summary and its
-- embedding vector (1024-dim, bge-large-en). On each new session start, the
-- top-K most semantically similar summaries are retrieved via pgvector and
-- injected as system context before the first Claude call.
CREATE TABLE IF NOT EXISTS long_term_memory (
    user_id    UUID         NOT NULL REFERENCES users(user_id)       ON DELETE CASCADE,
    session_id UUID         NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    summary    TEXT         NOT NULL,
    embedding  vector(1024) NOT NULL,
    created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, session_id)
);

-- ── Indexes ───────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_sessions_user    ON sessions        (user_id);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages        (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_user    ON messages        (user_id,    created_at);

-- HNSW index for fast approximate nearest-neighbour search on embeddings
CREATE INDEX IF NOT EXISTS idx_ltm_embedding    ON long_term_memory
    USING hnsw (embedding vector_cosine_ops);
