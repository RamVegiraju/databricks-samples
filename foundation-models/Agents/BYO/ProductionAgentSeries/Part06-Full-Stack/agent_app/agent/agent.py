"""
agent.py — LangGraph ReAct agent wiring all Part06 components together.

Tool surface:
  - MCP tools (add, return_biodata)   loaded at runtime from the deployed
                                       mcp_server/ Databricks App via
                                       DatabricksMultiServerMCPClient
  - get_product_info                  inline Python tool defined in tools.py
  - save_memory / get_memories        long-term memory tools from memory.py
                                       (LangGraph injects DatabricksStore at
                                       call time via InjectedStore)

Memory:
  - DatabricksStore  (long-term)      semantic search across sessions,
                                       namespaced per user_id
  - CheckpointSaver  (short-term)     full graph state persisted per
                                       thread_id after every step

The agent is initialized lazily on first request (async double-checked
locking) because DatabricksMultiServerMCPClient.get_tools() is async.
"""

import asyncio
import logging
import os

from databricks.sdk import WorkspaceClient
from databricks_langchain import (
    ChatDatabricks,
    DatabricksMCPServer,
    DatabricksMultiServerMCPClient,
)
from langgraph.prebuilt import create_react_agent

from .memory import MEMORY_TOOLS, get_checkpointer, get_store
from .tools import get_product_info

logger = logging.getLogger(__name__)

LLM_ENDPOINT = os.getenv("AGENT_LLM_ENDPOINT", "databricks-claude-sonnet-4-6")
LLM_TEMPERATURE = float(os.getenv("AGENT_LLM_TEMPERATURE", "0.1"))
MCP_SERVER_URL = os.environ["MCP_SERVER_URL"]

SYSTEM_PROMPT = """You are a helpful, knowledgeable assistant.

Memory guidelines:
- At the start of each conversation, call get_memories to retrieve relevant \
context about this user before responding.
- When the user shares important facts, preferences, or decisions, call \
save_memory to persist them for future sessions.
- Use retrieved memories to personalize your responses.

Tool use policy:
- Use get_product_info for Databricks product catalog queries.
- Use the add tool (MCP) for arithmetic operations.
- Use return_biodata (MCP) for person profile lookups.
- Only call tools when necessary to answer accurately.
- After tool results, incorporate the output into a clear final response.
""".strip()

# ── Lazy async initialization ─────────────────────────────────────────────────

_agent = None
_lock = asyncio.Lock()


async def _build_agent():
    """
    Builds the LangGraph agent once:
      1. Initializes DatabricksStore + CheckpointSaver (calls setup() on each)
      2. Connects to the remote MCP server and loads its tools
      3. Compiles a create_react_agent with all tools + store + checkpointer
    """
    ws = WorkspaceClient()

    # Long-term memory (cross-session, semantic search)
    store = get_store()
    store.setup()

    # Short-term memory (within-session state checkpointing)
    checkpointer = get_checkpointer()
    checkpointer.setup()

    # Remote MCP tools from the deployed mcp_server/ app
    mcp_client = DatabricksMultiServerMCPClient(
        [
            DatabricksMCPServer(
                name="production_mcp",
                url=MCP_SERVER_URL,
                workspace_client=ws,
            )
        ]
    )
    mcp_tools = await mcp_client.get_tools()
    logger.info("Loaded MCP tools: %s", [t.name for t in mcp_tools])

    # Full tool list: MCP tools + inline Python tool + memory tools
    all_tools = mcp_tools + [get_product_info] + MEMORY_TOOLS

    llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=LLM_TEMPERATURE)

    graph = create_react_agent(
        model=llm,
        tools=all_tools,
        store=store,
        checkpointer=checkpointer,
        prompt=SYSTEM_PROMPT,
    )
    return graph


async def get_agent():
    """Returns the compiled agent, initializing it on first call."""
    global _agent
    if _agent is None:
        async with _lock:
            if _agent is None:
                _agent = await _build_agent()
    return _agent
