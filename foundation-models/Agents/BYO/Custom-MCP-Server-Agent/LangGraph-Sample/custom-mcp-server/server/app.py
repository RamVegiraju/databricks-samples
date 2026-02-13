from fastapi import FastAPI, Request
from fastmcp import FastMCP

from .tools import load_tools
from .utils import header_store

# MCP server instance (Databricks template style)
mcp_server = FastMCP(name="custom-mcp-server")

# Register tools
load_tools(mcp_server)

# Convert MCP server to an HTTP app (FastMCP provides this)
mcp_app = mcp_server.http_app()

# Custom API app for extra endpoints
app = FastAPI(
    title="Custom MCP Server",
    version="0.1.0",
    lifespan=mcp_app.lifespan,
)

@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"ok": True}

# Combine MCP routes + your routes
combined_app = FastAPI(
    title="Combined MCP App",
    routes=[*mcp_app.routes, *app.routes],
    lifespan=mcp_app.lifespan,
)

@combined_app.middleware("http")
async def capture_headers(request: Request, call_next):
    # Needed for OBO auth inside Databricks Apps (x-forwarded-access-token)
    header_store.set(dict(request.headers))
    return await call_next(request)
