# Custom MCP Server for Databricks Apps

This directory contains a sample Model Context Protocol (MCP) server that can be developed locally with `uv` and deployed as a Databricks app running on the built-in HTTP transport. The server lives in `custom-mcp-server/` and exposes tools defined in `server/tools.py` alongside a simple `/healthz` route for smoke testing.

## Prerequisites

- Python 3.10 or later.
- [`uv`](https://docs.astral.sh/uv/) installed locally (`pip install uv` or the official install script).
- Databricks CLI authenticated against the target workspace (`databricks --version` to confirm availability).
- Workspace permissions to create and deploy Databricks apps.

## Local development

```bash
cd apps/mcp/custom-mcp-server

# Install and lock dependencies defined in pyproject.toml/requirements.txt
uv sync

# Run the server locally (listens on port 8000 by default)
uv run custom-server

# Optional: verify the health endpoint
curl http://127.0.0.1:8000/healthz
```

## Local MCP client tests

With the local server running on port `8000`, test tools via the local MCP client script:

```bash
cd apps/mcp
python3 test_local_mcp_client.py
```

This script calls:
- `add` with `{"a": 2, "b": 3}`
- `return_biodata` with `{"name": "Ram"}`

## First-time Databricks app deployment

```bash
cd apps/mcp/custom-mcp-server

# Log in to the workspace that will host the Databricks app
databricks auth login --host https://<your-workspace-hostname>

# (Once) create the Databricks app container
databricks apps create custom-mcp-server

# Resolve your workspace user folder for syncing sources
DATABRICKS_USERNAME=$(databricks current-user me | jq -r .userName)

# Upload source code to the workspace
databricks sync . "/Users/$DATABRICKS_USERNAME/custom-mcp-server"

# Deploy the MCP server app (app.yaml runs `uv run custom-server`)
databricks apps deploy custom-mcp-server \
  --source-code-path "/Workspace/Users/$DATABRICKS_USERNAME/custom-mcp-server"
```

After deployment completes, grab the application URL from the Databricks UI. The MCP transport endpoint will be available at `https://<app-hostname>/mcp`, and the health check lives at `https://<app-hostname>/healthz`.

## Updating after code changes

For subsequent iterations, do both steps in order:
1) `databricks sync` to upload changed files
2) `databricks apps deploy` to rebuild/restart the app with the updated source

```bash
cd apps/mcp/custom-mcp-server
databricks sync . "/Users/$DATABRICKS_USERNAME/custom-mcp-server"
databricks apps deploy custom-mcp-server \
  --source-code-path "/Workspace/Users/$DATABRICKS_USERNAME/custom-mcp-server"
```

`sync` alone updates workspace files, but the running app may not pick up those changes reliably until `deploy` is run.

Example:

```bash
cd apps/mcp/custom-mcp-server
databricks sync . "/Users/$DATABRICKS_USERNAME/custom-mcp-server"
databricks apps deploy custom-mcp-server \
  --source-code-path "/Workspace/Users/$DATABRICKS_USERNAME/custom-mcp-server"
```

If your Databricks app is already running, the deployment will roll forward in place. Use `databricks apps logs custom-mcp-server` for troubleshooting.

## References

- [Databricks MCP server hello world template](https://github.com/databricks/app-templates/tree/main/mcp-server-hello-world)
- [Databricks documentation: Host custom MCP servers using Databricks apps](https://docs.databricks.com/aws/en/generative-ai/mcp/custom-mcp#set-up-your-environment)

