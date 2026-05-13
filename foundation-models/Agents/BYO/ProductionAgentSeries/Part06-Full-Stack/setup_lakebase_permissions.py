"""
setup_lakebase_permissions.py — Grant the agent app's service principal the
permissions it needs on Lakebase.

Uses raw psycopg2 with SDK-generated OAuth tokens (Part04 pattern) — no
dependency on databricks-langchain.

Steps:
  1. Resolve the app's service principal client ID (UUID)
  2. Connect to Lakebase as admin via psycopg2 + short-lived SDK token
  3. Create the SP's Postgres role (idempotent)
  4. Enable pgvector extension
  5. Grant USAGE + CREATE on schema public
  6. Grant ALL on any existing tables in schema public
  7. ALTER DEFAULT PRIVILEGES so SP automatically gets access to future tables

Tables are created on first request by AsyncDatabricksStore.setup() and
AsyncCheckpointSaver.setup() — both are called per-request in server.py.
Since the SP has CREATE on the schema, it can create its own tables and will
own them automatically.

Requirements:
    pip install psycopg2-binary databricks-sdk python-dotenv

Usage:
    python setup_lakebase_permissions.py
    python setup_lakebase_permissions.py --app-name part06-agent-app --instance part06-agent-memory
"""

import argparse
import os
import uuid

import psycopg2
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "agent_app/.env"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grant SP permissions on Lakebase")
    p.add_argument("--app-name", default="part06-agent-app",   help="Databricks App name")
    p.add_argument("--instance", default="part06-agent-memory", help="Lakebase instance name")
    return p.parse_args()


def get_sp_client_id(w: WorkspaceClient, app_name: str) -> str:
    """Return the SP's client ID (UUID) — the Postgres role name used by Lakebase."""
    app = w.apps.get(app_name)

    sp_client_id = getattr(app, "service_principal_client_id", None)
    if not sp_client_id:
        sp = w.service_principals.get(app.service_principal_id)
        sp_client_id = str(sp.application_id)

    print(f"  SP display name : {getattr(app, 'service_principal_name', 'n/a')}")
    print(f"  SP client ID    : {sp_client_id}  <- Postgres role name")
    return sp_client_id


def _get_lakebase_conn(w: WorkspaceClient, instance_name: str) -> psycopg2.extensions.connection:
    """Open a psycopg2 connection to Lakebase using a short-lived SDK token (Part04 pattern)."""
    instance = w.database.get_database_instance(name=instance_name)
    host = instance.read_write_dns
    cred = w.database.generate_database_credential(
        request_id=str(uuid.uuid4()),
        instance_names=[instance_name],
    )
    conn = psycopg2.connect(
        host=host,
        port=5432,
        dbname="databricks_postgres",
        user=w.current_user.me().user_name,
        password=cred.token,
        sslmode="require",
        connect_timeout=10,
    )
    conn.autocommit = True
    return conn


def grant_sp_permissions(w: WorkspaceClient, instance_name: str, sp_client_id: str) -> None:
    """Create the SP Postgres role and grant schema + table permissions."""
    conn = _get_lakebase_conn(w, instance_name)
    try:
        with conn.cursor() as cur:
            # Create role for the SP (ignore if already exists)
            print(f"  Creating Postgres role for SP '{sp_client_id}' ...")
            cur.execute(
                f"""DO $$ BEGIN
                    CREATE ROLE "{sp_client_id}" WITH LOGIN;
                EXCEPTION WHEN duplicate_object THEN NULL;
                END $$;"""
            )

            # Enable pgvector extension (needed by store_vectors table)
            print("  Enabling pgvector extension ...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Grant USAGE + CREATE on schema public so SP can create its own tables
            print("  Granting USAGE, CREATE on schema public ...")
            cur.execute(f'GRANT USAGE, CREATE ON SCHEMA public TO "{sp_client_id}";')

            # Grant on any tables that already exist
            cur.execute("""
                SELECT tablename FROM pg_tables WHERE schemaname = 'public'
            """)
            existing = [r[0] for r in cur.fetchall()]
            if existing:
                print(f"  Granting ALL on {len(existing)} existing tables ...")
                cur.execute(
                    f'GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO "{sp_client_id}";'
                )

            # ALTER DEFAULT PRIVILEGES so SP gets access to any future tables created by admin
            print("  Setting default privileges for future tables ...")
            cur.execute(
                f'ALTER DEFAULT PRIVILEGES IN SCHEMA public '
                f'GRANT ALL ON TABLES TO "{sp_client_id}";'
            )
            cur.execute(
                f'ALTER DEFAULT PRIVILEGES IN SCHEMA public '
                f'GRANT ALL ON SEQUENCES TO "{sp_client_id}";'
            )

    finally:
        conn.close()

    print("  Permissions granted.")


def main() -> None:
    args = parse_args()

    w = WorkspaceClient(
        host=os.environ["DATABRICKS_HOST"],
        token=os.environ["DATABRICKS_TOKEN"],
    )

    print(f"\n[1/2] Resolving SP for app '{args.app_name}' ...")
    sp_client_id = get_sp_client_id(w, args.app_name)

    print(f"\n[2/2] Granting permissions to SP on '{args.instance}' ...")
    grant_sp_permissions(w, args.instance, sp_client_id)

    print("\nDone. The agent app's SP can now connect and create memory tables.")
    print("Tables will be created on first request via store.setup() / checkpointer.setup().")


if __name__ == "__main__":
    main()
