# test_remote_dbx_client.py
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient

CUSTOM_MCP_APP_URL = "Enter MCP App URL here + /mcp"

def main():
    ws = WorkspaceClient(profile="oauth")
    client = DatabricksMCPClient(
        server_url=CUSTOM_MCP_APP_URL,
        workspace_client=ws,
    )

    print("TOOLS:", client.list_tools())
    print("ADD:", client.call_tool("add", {"a": 2, "b": 3}))
    print("BIO:", client.call_tool("return_biodata", {"name": "Alice"}))

if __name__ == "__main__":
    main()
