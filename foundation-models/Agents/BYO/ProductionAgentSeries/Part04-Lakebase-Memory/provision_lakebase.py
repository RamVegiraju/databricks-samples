"""
Programmatically provision a Databricks Lakebase (managed PostgreSQL) instance
using the Databricks Python SDK, then bootstrap the chatbot schema.

Requirements:
    pip install databricks-sdk>=0.61.0 psycopg2-binary python-dotenv

Usage:
    python provision_lakebase.py
    python provision_lakebase.py --name my-chatbot-db --capacity CU_2 --retention 14
"""

import argparse
import os
import uuid

import psycopg2
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import DatabaseInstance
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# DDL – same schema used by memory.py
# ---------------------------------------------------------------------------
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    title        TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS messages (
    message_id   TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role         TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content      TEXT NOT NULL,
    token_count  INTEGER,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_session_time
    ON messages (session_id, created_at);

CREATE OR REPLACE VIEW session_summary AS
SELECT
    s.session_id,
    s.title,
    s.created_at,
    s.updated_at,
    COUNT(m.message_id)                                    AS total_messages,
    SUM(CASE WHEN m.role = 'user'      THEN 1 ELSE 0 END) AS user_turns,
    SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) AS assistant_turns,
    COALESCE(SUM(m.token_count), 0)                        AS total_tokens
FROM sessions s
LEFT JOIN messages m USING (session_id)
GROUP BY s.session_id, s.title, s.created_at, s.updated_at
ORDER BY s.updated_at DESC;
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Provision a Lakebase instance for the chatbot")
    p.add_argument("--name",      default="chatbot-memory",  help="Instance name (letters/hyphens, max 63 chars)")
    p.add_argument("--capacity",  default="CU_1",            help="Compute unit size: CU_1 | CU_2 | CU_4 …")
    p.add_argument("--retention", default=7, type=int,       help="Point-in-time recovery window in days (2-35)")
    p.add_argument("--database",  default="chatbot_memory",  help="Postgres database name to create")
    p.add_argument("--skip-schema", action="store_true",     help="Skip running DDL after provisioning")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1 – Create the Lakebase instance
# ---------------------------------------------------------------------------

def create_instance(w: WorkspaceClient, name: str, capacity: str, retention: int) -> DatabaseInstance:
    print(f"[1/3] Creating Lakebase instance '{name}' ({capacity}) …")

    instance = w.database.create_database_instance_and_wait(
        DatabaseInstance(
            name=name,
            capacity=capacity,
            retention_window_in_days=retention,
        )
    )

    print(f"      ✓ Instance ready")
    print(f"        Read/Write DNS : {instance.read_write_dns}")
    print(f"        Read-only  DNS : {getattr(instance, 'read_only_dns', 'n/a')}")
    return instance


# ---------------------------------------------------------------------------
# Step 2 – Generate a short-lived OAuth credential via the SDK
# ---------------------------------------------------------------------------

def get_credentials(w: WorkspaceClient, instance_name: str) -> tuple[str, str]:
    """Returns (pg_user, pg_password) using SDK-generated token auth."""
    print("[2/3] Generating database credentials …")

    # The SDK generates a short-lived OAuth token scoped to the instance.
    # Use your Databricks user name as the PG user.
    pg_user = w.current_user.me().user_name
    cred = w.database.generate_database_credential(
        request_id=str(uuid.uuid4()),
        instance_names=[instance_name],
    )
    print(f"      ✓ Credential generated for user '{pg_user}'")
    return pg_user, cred.token


# ---------------------------------------------------------------------------
# Step 3 – Bootstrap the schema
# ---------------------------------------------------------------------------

def bootstrap_schema(host: str, pg_user: str, pg_password: str, database: str) -> None:
    print(f"[3/3] Bootstrapping schema in database '{database}' …")

    # The default database on a new Lakebase instance is 'databricks_postgres'.
    # Connect to it first and create our target database if needed.
    admin_conn = psycopg2.connect(
        host=host,
        port=5432,
        dbname="databricks_postgres",
        user=pg_user,
        password=pg_password,
        sslmode="require",
        connect_timeout=15,
    )
    admin_conn.autocommit = True
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (database,)
        )
        if not cur.fetchone():
            cur.execute(f'CREATE DATABASE "{database}"')
            print(f"      ✓ Database '{database}' created")
        else:
            print(f"      ✓ Database '{database}' already exists")
    admin_conn.close()

    # Now connect to the target database and apply DDL.
    app_conn = psycopg2.connect(
        host=host,
        port=5432,
        dbname=database,
        user=pg_user,
        password=pg_password,
        sslmode="require",
        connect_timeout=15,
    )
    app_conn.autocommit = False
    with app_conn.cursor() as cur:
        cur.execute(_SCHEMA_SQL)
    app_conn.commit()
    app_conn.close()
    print("      ✓ Schema applied (sessions, messages, session_summary view)")


# ---------------------------------------------------------------------------
# Print .env values for the user to copy
# ---------------------------------------------------------------------------

def print_env_snippet(instance, pg_user: str, database: str) -> None:
    print("\n" + "=" * 60)
    print("Add these to your .env file:")
    print("=" * 60)
    print(f"LAKEBASE_HOST={instance.read_write_dns}")
    print(f"LAKEBASE_PORT=5432")
    print(f"LAKEBASE_DATABASE={database}")
    print(f"LAKEBASE_USER={pg_user}")
    print(f"LAKEBASE_PASSWORD=<generate fresh token at runtime via SDK>")
    print(f"LAKEBASE_SSLMODE=require")
    print("=" * 60)
    print()
    print("Note: Lakebase uses short-lived OAuth tokens.")
    print("      For production, call w.database.generate_database_credential()")
    print("      at app startup and refresh before expiry (~1 hour).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # WorkspaceClient auto-reads DATABRICKS_HOST + DATABRICKS_TOKEN from env.
    w = WorkspaceClient(
        host=os.environ["DATABRICKS_HOST"],
        token=os.environ["DATABRICKS_TOKEN"],
    )

    instance = create_instance(w, args.name, args.capacity, args.retention)
    pg_user, pg_password = get_credentials(w, args.name)

    if not args.skip_schema:
        bootstrap_schema(instance.read_write_dns, pg_user, pg_password, args.database)

    print_env_snippet(instance, pg_user, args.database)
    print("Done! Your Lakebase instance is ready for the chatbot.")


if __name__ == "__main__":
    main()
