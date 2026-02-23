"""
db.py – Lakebase (PostgreSQL) helper layer.

Credentials are generated at runtime via the Databricks SDK (short-lived
OAuth tokens, ~1 hour TTL). The connection is transparently refreshed when
the token is within 5 minutes of expiry — no static passwords needed.

Usage:
    from db import Database

    db = Database()
    user_id    = db.get_or_create_user("alice")
    session_id = db.start_session(user_id)

    db.add_message(session_id, user_id, "user",      "Hello!")
    db.add_message(session_id, user_id, "assistant", "Hi there!", token_count=12)

    history = db.get_session_history(session_id)   # → [{role, content}, ...]
    db.end_session(session_id)
    db.close()
"""

import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv

load_dotenv()

# Refresh the token this many minutes before it actually expires.
_REFRESH_BUFFER_MINUTES = 5
# Databricks Lakebase OAuth tokens are valid for ~60 minutes.
_TOKEN_TTL_MINUTES = 60


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Database:
    def __init__(self):
        self._workspace = WorkspaceClient(
            host=os.environ["DATABRICKS_HOST"],
            token=os.environ["DATABRICKS_TOKEN"],
        )
        self._instance_name = os.environ["LAKEBASE_INSTANCE_NAME"]
        self._host     = os.environ["LAKEBASE_HOST"]
        self._port     = int(os.environ.get("LAKEBASE_PORT", 5432))
        self._dbname   = os.environ["LAKEBASE_DATABASE"]
        self._user     = os.environ["LAKEBASE_USER"]
        self._sslmode  = os.environ.get("LAKEBASE_SSLMODE", "require")

        self._conn: Optional[psycopg2.extensions.connection] = None
        self._token_expires_at: Optional[datetime] = None

        # Connect immediately on construction
        self._connect()

    # ── Credential + connection management ──────────────────────────────────

    def _generate_token(self) -> str:
        """Call the Databricks SDK to get a fresh short-lived OAuth token."""
        cred = self._workspace.database.generate_database_credential(
            request_id=str(uuid.uuid4()),
            instance_names=[self._instance_name],
        )
        self._token_expires_at = _utcnow() + timedelta(minutes=_TOKEN_TTL_MINUTES)
        return cred.token

    def _connect(self) -> None:
        """Open a new psycopg2 connection using a freshly generated token."""
        if self._conn and not self._conn.closed:
            self._conn.close()

        token = self._generate_token()
        self._conn = psycopg2.connect(
            host=self._host,
            port=self._port,
            dbname=self._dbname,
            user=self._user,
            password=token,
            sslmode=self._sslmode,
            connect_timeout=10,
        )
        self._conn.autocommit = False
        print(f"[db] Connected. Token valid until {self._token_expires_at.strftime('%H:%M:%S UTC')}")

    def _refresh_if_needed(self) -> None:
        """Reconnect with a new token if we're close to expiry or disconnected."""
        if self._conn is None or self._conn.closed:
            self._connect()
            return

        if self._token_expires_at is None:
            self._connect()
            return

        time_left = self._token_expires_at - _utcnow()
        if time_left <= timedelta(minutes=_REFRESH_BUFFER_MINUTES):
            print("[db] Token nearing expiry — refreshing connection …")
            self._connect()

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Internal ────────────────────────────────────────────────────────────

    def _execute(self, sql: str, params: tuple = ()) -> list[dict]:
        """Refresh token if needed, run statement, return rows as dicts."""
        self._refresh_if_needed()
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            self._conn.commit()
            try:
                return [dict(r) for r in cur.fetchall()]
            except psycopg2.ProgrammingError:
                return []  # INSERT/UPDATE with no RETURNING clause

    # ── Users ────────────────────────────────────────────────────────────────

    def get_or_create_user(self, username: str) -> str:
        """
        Return the user_id for *username*, creating the row if it doesn't exist.
        This is the entry point when a user logs in.
        """
        rows = self._execute(
            """
            INSERT INTO users (username)
            VALUES (%s)
            ON CONFLICT (username) DO UPDATE SET username = EXCLUDED.username
            RETURNING user_id
            """,
            (username,),
        )
        return str(rows[0]["user_id"])

    def get_user(self, username: str) -> Optional[dict]:
        """Look up a user by username. Returns None if not found."""
        rows = self._execute(
            "SELECT user_id, username, created_at FROM users WHERE username = %s",
            (username,),
        )
        return rows[0] if rows else None

    # ── Sessions ─────────────────────────────────────────────────────────────

    def start_session(self, user_id: str) -> str:
        """
        Insert a new session row for *user_id*.
        Call this every time a user logs in.
        Returns the new session_id.
        """
        rows = self._execute(
            "INSERT INTO sessions (user_id) VALUES (%s) RETURNING session_id",
            (user_id,),
        )
        return str(rows[0]["session_id"])

    def end_session(self, session_id: str) -> None:
        """Mark a session as ended. Call on logout or timeout."""
        self._execute(
            "UPDATE sessions SET ended_at = NOW() WHERE session_id = %s",
            (session_id,),
        )

    def get_active_sessions(self, user_id: str) -> list[dict]:
        """Return all sessions where ended_at IS NULL for a user."""
        return self._execute(
            """
            SELECT session_id, started_at
            FROM sessions
            WHERE user_id = %s AND ended_at IS NULL
            ORDER BY started_at DESC
            """,
            (user_id,),
        )

    def get_user_sessions(self, user_id: str, limit: int = 20) -> list[dict]:
        """Return all sessions (active and closed) for a user, newest first."""
        return self._execute(
            """
            SELECT session_id, started_at, ended_at,
                   ended_at IS NOT NULL AS is_closed
            FROM sessions
            WHERE user_id = %s
            ORDER BY started_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        )

    # ── Messages ─────────────────────────────────────────────────────────────

    def add_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        token_count: Optional[int] = None,
    ) -> str:
        """
        Persist a single turn. Call twice per round-trip:
            add_message(..., "user",      user_text,  token_count=prompt_tokens)
            add_message(..., "assistant", reply_text, token_count=completion_tokens)
        Returns the new message_id.
        """
        rows = self._execute(
            """
            INSERT INTO messages (session_id, user_id, role, content, token_count)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING message_id
            """,
            (session_id, user_id, role, content, token_count),
        )
        return str(rows[0]["message_id"])

    def get_session_history(self, session_id: str, limit: int = 40) -> list[dict]:
        """
        Return the last *limit* messages for a session in chronological order,
        formatted as [{role, content}] — ready to pass straight to the FM API.
        """
        rows = self._execute(
            """
            SELECT role, content
            FROM (
                SELECT role, content, created_at
                FROM messages
                WHERE session_id = %s
                ORDER BY created_at DESC
                LIMIT %s
            ) sub
            ORDER BY created_at ASC
            """,
            (session_id, limit),
        )
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def get_user_history(self, user_id: str, limit: int = 100) -> list[dict]:
        """
        All messages for a user across every session, newest first.
        Useful for analytics or cross-session context.
        """
        return self._execute(
            """
            SELECT m.message_id, m.session_id, m.role, m.content,
                   m.token_count, m.created_at
            FROM messages m
            WHERE m.user_id = %s
            ORDER BY m.created_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        )

    # ── Long-term memory ─────────────────────────────────────────────────────

    def save_long_term_memory(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        embedding: list,
    ) -> None:
        """
        Upsert a session summary + its embedding vector into long_term_memory.
        Embedding is passed as a Python list and cast to vector in SQL.
        """
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
        self._execute(
            """
            INSERT INTO long_term_memory (user_id, session_id, summary, embedding)
            VALUES (%s, %s, %s, %s::vector)
            ON CONFLICT (user_id, session_id)
            DO UPDATE SET summary = EXCLUDED.summary, embedding = EXCLUDED.embedding
            """,
            (user_id, session_id, summary, embedding_str),
        )

    def get_relevant_memories(
        self, user_id: str, query_embedding: list, k: int = 5
    ) -> list[str]:
        """
        Return the top-K most semantically similar session summaries for a user,
        ranked by cosine distance to query_embedding via pgvector.
        """
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        rows = self._execute(
            """
            SELECT summary
            FROM long_term_memory
            WHERE user_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (user_id, embedding_str, k),
        )
        return [r["summary"] for r in rows]

    # ── Stats ────────────────────────────────────────────────────────────────

    def get_session_stats(self, session_id: str) -> dict:
        rows = self._execute(
            """
            SELECT
                COUNT(*)                                                AS total_messages,
                SUM(CASE WHEN role = 'user'      THEN 1 ELSE 0 END)    AS user_turns,
                SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END)    AS assistant_turns,
                COALESCE(SUM(token_count), 0)                          AS total_tokens
            FROM messages
            WHERE session_id = %s
            """,
            (session_id,),
        )
        return rows[0]
