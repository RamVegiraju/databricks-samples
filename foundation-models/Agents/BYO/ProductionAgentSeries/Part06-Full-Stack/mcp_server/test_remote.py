"""
test_remote.py — Validate the deployed Part06 MCP server app and its tools.

Usage:
    cd mcp_server/
    python test_remote.py
"""

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient

APP_MCP_URL = "https://part06-mcp-server-7474655512364318.aws.databricksapps.com/mcp"


def main():
    ws = WorkspaceClient(profile="oauth")
    client = DatabricksMCPClient(server_url=APP_MCP_URL, workspace_client=ws)

    print("=== Available tools ===")
    print(client.list_tools())

    print("\n=== add(7, 8) ===")
    print(client.call_tool("add", {"a": 7, "b": 8}))

    print("\n=== return_biodata('Alice') ===")
    print(client.call_tool("return_biodata", {"name": "Alice"}))


if __name__ == "__main__":
    main()
