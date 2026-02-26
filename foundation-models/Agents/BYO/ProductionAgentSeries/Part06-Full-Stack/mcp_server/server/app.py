from fastapi import FastAPI, Request
from fastmcp import FastMCP

from .tools import load_tools
from .utils import header_store

# MCP server instance
mcp_server = FastMCP(name="production-mcp-server")

# Register tools
load_tools(mcp_server)

# Convert MCP server to an HTTP app
mcp_app = mcp_server.http_app()

# Custom API app for extra endpoints
app = FastAPI(
    title="Production MCP Server",
    version="0.1.0",
    lifespan=mcp_app.lifespan,
)


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"ok": True}


# Combine MCP routes + custom routes
combined_app = FastAPI(
    title="Combined MCP App",
    routes=[*mcp_app.routes, *app.routes],
    lifespan=mcp_app.lifespan,
)


@combined_app.middleware("http")
async def capture_headers(request: Request, call_next):
    # Capture forwarded access token for OBO auth inside Databricks Apps
    header_store.set(dict(request.headers))
    return await call_next(request)
