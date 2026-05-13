"""
agent.py — LangGraph ReAct agent wiring all Part06 components together.

Tool surface:
  - MCP tools (add, return_biodata)   loaded at runtime from the deployed
                                       mcp_server/ Databricks App via
                                       DatabricksMultiServerMCPClient
  - get_product_info                  inline Python tool defined in tools.py
  - get_user_memory / save_user_memory / delete_user_memory
                                       long-term memory tools from memory.py
                                       (store injected via RunnableConfig)

Memory:
  - AsyncDatabricksStore  (long-term)  semantic search across sessions,
                                       namespaced per user_id. Embedding model
                                       (DATABRICKS_EMBEDDING_ENDPOINT) is used
                                       to embed both stored memories and queries.
  - AsyncCheckpointSaver  (short-term) full graph state persisted per
                                       thread_id after every step.

Both store and checkpointer are used as async context managers per-request
(following app-templates best practices). The agent is initialized on every
request — this keeps connection lifecycle correct and the code simple.

Reference: https://github.com/databricks/app-templates/tree/main/agent-langgraph-long-term-memory
"""

import logging
import os

from databricks.sdk import WorkspaceClient
from databricks_langchain import (
    ChatDatabricks,
    DatabricksMultiServerMCPClient,
    MCPServer,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.base import BaseStore

from .memory import memory_tools, resolve_lakebase_instance_name
from .tools import get_product_info

logger = logging.getLogger(__name__)

LLM_ENDPOINT = os.getenv("AGENT_LLM_ENDPOINT", "databricks-gpt-oss-120b")
LLM_TEMPERATURE = float(os.getenv("AGENT_LLM_TEMPERATURE", "0.1"))
MCP_SERVER_URL = os.environ["MCP_SERVER_URL"]

_LAKEBASE_INSTANCE_NAME_RAW = os.getenv("LAKEBASE_INSTANCE_NAME", "")
if not _LAKEBASE_INSTANCE_NAME_RAW:
    raise ValueError(
        "LAKEBASE_INSTANCE_NAME environment variable is required but not set. "
        "Set it in app.yaml or agent_app/.env."
    )
LAKEBASE_INSTANCE_NAME = resolve_lakebase_instance_name(_LAKEBASE_INSTANCE_NAME_RAW)

SYSTEM_PROMPT = """You are a helpful, knowledgeable assistant.

Memory guidelines:
- At the start of each conversation, call get_user_memory to retrieve relevant \
context about this user before responding.
- When the user shares important facts, preferences, or decisions, call \
save_user_memory to persist them for future sessions.
- If the user asks you to forget something, call delete_user_memory.
- Use retrieved memories to personalize your responses.

Tool use policy:
- Use get_product_info for Databricks product catalog queries.
- Use the add tool (MCP) for arithmetic operations.
- Use return_biodata (MCP) for person profile lookups.
- Only call tools when necessary to answer accurately.
- After tool results, incorporate the output into a clear final response.
""".strip()

# Global service-principal workspace client (safe to share across requests)
sp_workspace_client = WorkspaceClient()


async def init_agent(
    store: BaseStore,
    checkpointer: BaseCheckpointSaver,
    workspace_client: WorkspaceClient | None = None,
):
    """Build a LangGraph ReAct agent for a single request.

    Called once per request inside the async with store/checkpointer context.
    Connects to the MCP server, loads tools, and compiles the graph.
    """
    ws = workspace_client or sp_workspace_client

    # Use MCPServer with explicit auth headers instead of DatabricksMCPServer.
    # DatabricksMCPServer passes server_url="" to OAuthClientProvider, which
    # breaks OAuth discovery (builds URLs from empty string) if the MCP server
    # returns 401. Passing the Bearer token directly avoids this issue.
    auth_headers = ws.config.authenticate()
    mcp_client = DatabricksMultiServerMCPClient(
        [
            MCPServer(
                name="production_mcp",
                url=MCP_SERVER_URL,
                headers=auth_headers,
            )
        ]
    )
    mcp_tools = await mcp_client.get_tools()
    logger.info("Loaded MCP tools: %s", [t.name for t in mcp_tools])

    all_tools = mcp_tools + [get_product_info] + memory_tools()

    llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=LLM_TEMPERATURE)

    return create_react_agent(
        model=llm,
        tools=all_tools,
        store=store,
        checkpointer=checkpointer,
        prompt=SYSTEM_PROMPT,
    )
