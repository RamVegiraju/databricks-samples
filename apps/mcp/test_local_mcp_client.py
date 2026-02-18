from databricks_mcp import DatabricksMCPClient


def main():
    # Local MCP endpoint exposed by custom-mcp-server.
    client = DatabricksMCPClient(server_url="http://127.0.0.1:8000/mcp")

    print("TOOLS:", client.list_tools())
    print("ADD (2 + 3):", client.call_tool("add", {"a": 2, "b": 3}))
    print("BIODATA (Ram):", client.call_tool("return_biodata", {"name": "Ram"}))


if __name__ == "__main__":
    main()
