"""
memory.py — Long-term memory + checkpointing via databricks_langchain.

Replaces the ~400-line custom db.py + memory.py from Part04 with two
SDK-managed abstractions:

  DatabricksStore    — cross-session long-term memory backed by Lakebase +
                       pgvector semantic search.  Implements LangGraph's
                       BaseStore interface.  Passed as store= to the agent.

  CheckpointSaver    — within-session short-term memory (full graph state
                       checkpointed to Postgres after every step).  Implements
                       LangGraph's BaseCheckpointSaver interface.  Passed as
                       checkpointer= to the agent.

Both handle connection pooling and OAuth token refresh automatically via the
Databricks SDK — no manual token management needed (contrast with Part04/db.py).

Memory tools (save_memory, get_memories) are injected into the agent's tool list
so the LLM can explicitly read/write long-term memories during a conversation.
"""

import os
import uuid
from typing import Annotated

from databricks_langchain import CheckpointSaver, DatabricksStore
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore


# ── Store + checkpointer factories ───────────────────────────────────────────

def get_store() -> DatabricksStore:
    """
    Returns a DatabricksStore backed by the configured Lakebase instance.
    Supports semantic search via the embedding endpoint (pgvector under the hood).
    Call store.setup() once before first use to create the required tables.
    """
    return DatabricksStore(
        instance_name=os.environ["LAKEBASE_INSTANCE_NAME"],
        embedding_endpoint=os.getenv("DATABRICKS_EMBEDDING_ENDPOINT", "databricks-bge-large-en"),
        embedding_dims=int(os.getenv("EMBEDDING_DIMS", "1024")),
    )


def get_checkpointer() -> CheckpointSaver:
    """
    Returns a CheckpointSaver (LangGraph PostgresSaver) backed by Lakebase.
    Persists full graph state per thread_id after every node execution,
    enabling seamless multi-turn conversations with state recovery.
    Call checkpointer.setup() once before first use.
    """
    return CheckpointSaver(
        instance_name=os.environ["LAKEBASE_INSTANCE_NAME"],
    )


# ── Memory tools — injected into the agent's tool list ───────────────────────
# InjectedStore and InjectedToolArg mark parameters as runtime-injected by
# LangGraph; they are hidden from the LLM's tool schema so the model never
# tries to fill them in.

@tool
def save_memory(
    content: str,
    config: Annotated[RunnableConfig, InjectedToolArg],
    store: Annotated[BaseStore, InjectedStore],
) -> str:
    """Save an important fact or user preference to long-term memory."""
    user_id = config.get("configurable", {}).get("user_id", "anonymous")
    store.put(("memories", user_id), str(uuid.uuid4()), {"content": content})
    return "Memory saved."


@tool
def get_memories(
    query: str,
    config: Annotated[RunnableConfig, InjectedToolArg],
    store: Annotated[BaseStore, InjectedStore],
) -> str:
    """Retrieve relevant long-term memories about the user based on a search query."""
    user_id = config.get("configurable", {}).get("user_id", "anonymous")
    results = store.search(("memories", user_id), query=query, limit=5)
    if not results:
        return "No relevant memories found."
    return "\n".join(f"- {r.value['content']}" for r in results)


MEMORY_TOOLS = [save_memory, get_memories]
