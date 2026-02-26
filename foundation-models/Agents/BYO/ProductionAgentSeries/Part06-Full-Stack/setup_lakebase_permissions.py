"""
setup_lakebase_permissions.py — Initialize Lakebase tables and grant the agent
app's service principal the permissions it needs.

Two-step process (following Databricks app-templates best practice):

  Step A — Initialize tables as admin:
    Runs store.setup() and checkpointer.setup() locally with admin credentials.
    This pre-creates all required tables (store, store_vectors, store_migrations,
    vector_migrations, checkpoints, checkpoint_writes, checkpoint_migrations) so
    the deployed SP only needs DML — not CREATE.
    Tables: store, store_vectors, store_migrations, vector_migrations,
            checkpoints, checkpoint_blobs, checkpoint_writes, checkpoint_migrations

  Step B — Grant permissions to SP via LakebaseClient:
    Uses databricks_ai_bridge.lakebase.LakebaseClient to:
      1. create_role() — register the SP's Postgres role
      2. grant_schema() — USAGE on public schema
      3. grant_table()  — SELECT/INSERT/UPDATE/DELETE on each table

Run this ONCE after app creation (Step 6b) and before testing the deployed app.

Requirements:
    pip install "databricks-langchain[memory]" databricks-sdk python-dotenv

Usage:
    python setup_lakebase_permissions.py
    python setup_lakebase_permissions.py --app-name part06-agent-app --instance part06-agent-memory
"""

import argparse
import os

from databricks.sdk import WorkspaceClient
from databricks_ai_bridge.lakebase import LakebaseClient, SchemaPrivilege, TablePrivilege
import asyncio

from databricks_langchain import AsyncCheckpointSaver, DatabricksStore
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "agent_app/.env"))

# Tables created by DatabricksStore.setup() and CheckpointSaver.setup()
STORE_TABLES = [
    "public.store",
    "public.store_vectors",
    "public.store_migrations",
    "public.vector_migrations",
]
CHECKPOINT_TABLES = [
    "public.checkpoints",
    "public.checkpoint_blobs",
    "public.checkpoint_writes",
    "public.checkpoint_migrations",
]
ALL_TABLES = STORE_TABLES + CHECKPOINT_TABLES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Initialize Lakebase tables and grant SP permissions")
    p.add_argument("--app-name",  default="part06-agent-app",    help="Databricks App name")
    p.add_argument("--instance",  default="part06-agent-memory",  help="Lakebase instance name")
    p.add_argument(
        "--skip-init", action="store_true",
        help="Skip table initialization (if tables already exist)"
    )
    return p.parse_args()


def get_sp_client_id(w: WorkspaceClient, app_name: str) -> str:
    """Return the SP's client ID (UUID) — the Postgres role name used by Lakebase."""
    app = w.apps.get(app_name)

    # Prefer service_principal_client_id (direct field) if available
    sp_client_id = getattr(app, "service_principal_client_id", None)
    if not sp_client_id:
        # Fall back: look up application_id via service_principals API
        sp = w.service_principals.get(app.service_principal_id)
        sp_client_id = str(sp.application_id)

    print(f"  SP display name : {getattr(app, 'service_principal_name', 'n/a')}")
    print(f"  SP client ID    : {sp_client_id}  ← Postgres role name")
    return sp_client_id


def init_tables(instance_name: str, embedding_endpoint: str, embedding_dims: int) -> None:
    """Run store.setup() and checkpointer.setup() as admin to pre-create all tables."""
    print("  Initializing DatabricksStore tables ...")
    store = DatabricksStore(
        instance_name=instance_name,
        embedding_endpoint=embedding_endpoint,
        embedding_dims=embedding_dims,
    )
    store.setup()
    print("  DatabricksStore tables ready.")

    print("  Initializing AsyncCheckpointSaver tables ...")
    asyncio.run(_init_checkpointer(instance_name))
    print("  AsyncCheckpointSaver tables ready.")


async def _init_checkpointer(instance_name: str) -> None:
    # AsyncCheckpointSaver requires a running event loop at instantiation time
    # and must be used as an async context manager to open its connection pool.
    async with AsyncCheckpointSaver(instance_name=instance_name) as checkpointer:
        await checkpointer.setup()


def grant_sp_permissions(instance_name: str, sp_client_id: str) -> None:
    """Use LakebaseClient to create the SP role and grant schema + table permissions."""
    client = LakebaseClient(instance_name=instance_name)

    print(f"  Creating Postgres role for SP '{sp_client_id}' ...")
    client.create_role(sp_client_id, "SERVICE_PRINCIPAL")

    print("  Granting USAGE on schema public ...")
    client.grant_schema(
        grantee=sp_client_id,
        schemas=["public"],
        privileges=[SchemaPrivilege.USAGE, SchemaPrivilege.CREATE],
    )

    print(f"  Granting DML on {len(ALL_TABLES)} tables ...")
    client.grant_table(
        grantee=sp_client_id,
        tables=ALL_TABLES,
        privileges=[
            TablePrivilege.SELECT,
            TablePrivilege.INSERT,
            TablePrivilege.UPDATE,
            TablePrivilege.DELETE,
        ],
    )
    print("  Permissions granted.")


def main() -> None:
    args = parse_args()

    instance = args.instance
    embedding_endpoint = os.getenv("DATABRICKS_EMBEDDING_ENDPOINT", "databricks-bge-large-en")
    embedding_dims = int(os.getenv("EMBEDDING_DIMS", "1024"))

    w = WorkspaceClient(
        host=os.environ["DATABRICKS_HOST"],
        token=os.environ["DATABRICKS_TOKEN"],
    )

    print(f"\n[1/3] Resolving SP for app '{args.app_name}' ...")
    sp_client_id = get_sp_client_id(w, args.app_name)

    if not args.skip_init:
        print(f"\n[2/3] Initializing Lakebase tables on '{instance}' as admin ...")
        init_tables(instance, embedding_endpoint, embedding_dims)
    else:
        print("\n[2/3] Skipping table initialization (--skip-init).")

    print(f"\n[3/3] Granting permissions to SP on '{instance}' ...")
    grant_sp_permissions(instance, sp_client_id)

    print("\nDone. The agent app can now read/write memory tables.")
    print("Retry the test — no redeploy needed.")


if __name__ == "__main__":
    main()
