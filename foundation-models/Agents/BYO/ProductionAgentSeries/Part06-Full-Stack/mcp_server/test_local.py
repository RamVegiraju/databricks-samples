"""
test_local.py — Validate the MCP server is running and tools are reachable.

Run the server first in a separate terminal:
    python server/main.py --port 8001

Then run this script:
    python test_local.py
"""

from databricks_mcp import DatabricksMCPClient


def main():
    client = DatabricksMCPClient(server_url="http://127.0.0.1:8001/mcp")

    print("=== Available tools ===")
    print(client.list_tools())

    print("\n=== add(7, 8) ===")
    print(client.call_tool("add", {"a": 7, "b": 8}))

    print("\n=== return_biodata('Alice') ===")
    print(client.call_tool("return_biodata", {"name": "Alice"}))


if __name__ == "__main__":
    main()
