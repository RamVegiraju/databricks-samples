"""
grant_mcp_permissions.py — Grant the agent app's service principal CAN_USE
permission on the MCP server app so it can call it from within Databricks Apps.

Usage:
    python grant_mcp_permissions.py
    python grant_mcp_permissions.py --agent-app part06-agent-app --mcp-app part06-mcp-server
"""

import argparse
import os

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.iam import AccessControlRequest, PermissionLevel
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "agent_app/.env"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grant agent app SP CAN_USE on MCP server app")
    p.add_argument("--agent-app", default="part06-agent-app", help="Agent app name")
    p.add_argument("--mcp-app",   default="part06-mcp-server", help="MCP server app name")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    w = WorkspaceClient(
        host=os.environ["DATABRICKS_HOST"],
        token=os.environ["DATABRICKS_TOKEN"],
    )

    # Get the agent app's service principal — try UUID (application_id) first
    agent_app = w.apps.get(args.agent_app)
    numeric_id = str(agent_app.service_principal_id)
    sp = w.service_principals.get(agent_app.service_principal_id)
    uuid_id = str(sp.application_id)
    sp_display = sp.display_name
    print(f"Agent app SP : {sp_display}")
    print(f"  numeric id : {numeric_id}")
    print(f"  UUID       : {uuid_id}")

    # Try both identifiers — update_permissions merges, does not replace
    for sp_id in [uuid_id, numeric_id]:
        print(f"\nTrying update_permissions with SP id='{sp_id}' ...")
        try:
            result = w.apps.update_permissions(
                args.mcp_app,
                access_control_list=[
                    AccessControlRequest(
                        service_principal_name=sp_id,
                        permission_level=PermissionLevel.CAN_USE,
                    )
                ],
            )
            print("  Success. Current ACL:")
            for entry in result.access_control_list:
                perms = [p.permission_level.value for p in (entry.all_permissions or [])]
                identity = (
                    getattr(entry, "service_principal_name", None)
                    or getattr(entry, "user_name", None)
                    or getattr(entry, "group_name", None)
                )
                print(f"    {identity}: {perms}")
        except Exception as e:
            print(f"  Failed: {e}")


if __name__ == "__main__":
    main()
