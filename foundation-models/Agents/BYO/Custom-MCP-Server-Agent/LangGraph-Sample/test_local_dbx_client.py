from databricks_mcp import DatabricksMCPClient

def main():
    # IMPORTANT: For your server, MCP is mounted at /mcp
    client = DatabricksMCPClient(server_url="http://127.0.0.1:8000/mcp")

    print("TOOLS:", client.list_tools())
    print("ADD:", client.call_tool("add", {"a": 2, "b": 3}))

if __name__ == "__main__":
    main()
