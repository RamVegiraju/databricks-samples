"""
Provision a Databricks Lakebase (managed PostgreSQL) instance for Part06.

This is simpler than the Part04 version — no custom schema DDL is needed here
because DatabricksStore and CheckpointSaver manage their own tables via
store.setup() / checkpointer.setup() at agent startup.  All we need is the
instance itself and its name.

Requirements:
    pip install databricks-sdk>=0.61.0 python-dotenv

Usage:
    python provision_lakebase.py
    python provision_lakebase.py --name part06-agent-memory --capacity CU_2
"""

import argparse
import os
import uuid

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import DatabaseInstance
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "agent_app/.env"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Provision a Lakebase instance for Part06")
    p.add_argument("--name",      default="part06-agent-memory", help="Instance name (letters/hyphens, max 63 chars)")
    p.add_argument("--capacity",  default="CU_1",                help="Compute unit size: CU_1 | CU_2 | CU_4 …")
    p.add_argument("--retention", default=7, type=int,           help="Point-in-time recovery window in days (2-35)")
    return p.parse_args()


# ── Step 1: Create the Lakebase instance ─────────────────────────────────────

def create_instance(w: WorkspaceClient, name: str, capacity: str, retention: int) -> DatabaseInstance:
    print(f"\n[1/2] Creating Lakebase instance '{name}' ({capacity}) ...")
    print("      This may take a few minutes.")

    instance = w.database.create_database_instance_and_wait(
        DatabaseInstance(
            name=name,
            capacity=capacity,
            retention_window_in_days=retention,
        )
    )

    print(f"      Instance ready")
    print(f"        Read/Write DNS : {instance.read_write_dns}")
    return instance


# ── Step 2: Verify credentials can be generated ──────────────────────────────

def verify_credentials(w: WorkspaceClient, instance_name: str) -> None:
    print(f"\n[2/2] Verifying database credentials ...")
    pg_user = w.current_user.me().user_name
    w.database.generate_database_credential(
        request_id=str(uuid.uuid4()),
        instance_names=[instance_name],
    )
    print(f"      Credentials verified for user '{pg_user}'")
    print(f"      Tokens are short-lived (~1 hour) and refreshed automatically")
    print(f"      by DatabricksStore / CheckpointSaver at runtime.")


# ── Print .env snippet ────────────────────────────────────────────────────────

def print_env_snippet(instance_name: str) -> None:
    print("\n" + "=" * 60)
    print("Add this to agent_app/.env (copy from .env.example):")
    print("=" * 60)
    print(f"LAKEBASE_INSTANCE_NAME={instance_name}")
    print("=" * 60)
    print()
    print("DatabricksStore and CheckpointSaver will call store.setup()")
    print("and checkpointer.setup() at agent startup to create their")
    print("own tables — no manual schema DDL required.")
    print()
    print("Next steps:")
    print("  1. Set LAKEBASE_INSTANCE_NAME in agent_app/.env")
    print("  2. Deploy mcp_server/  →  set MCP_SERVER_URL in .env")
    print("  3. Deploy agent_app/")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    w = WorkspaceClient(
        host=os.environ["DATABRICKS_HOST"],
        token=os.environ["DATABRICKS_TOKEN"],
    )

    instance = create_instance(w, args.name, args.capacity, args.retention)
    verify_credentials(w, args.name)
    print_env_snippet(args.name)
    print("Done! Lakebase instance is ready.")


if __name__ == "__main__":
    main()
