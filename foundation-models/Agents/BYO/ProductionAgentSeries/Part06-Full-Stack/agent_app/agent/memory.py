"""
memory.py — Long-term memory (AsyncDatabricksStore) + short-term checkpointing
(AsyncCheckpointSaver) following app-templates best practices.

Long-term memory:
  AsyncDatabricksStore — cross-session semantic search backed by Lakebase +
  pgvector. The configured embedding model (DATABRICKS_EMBEDDING_ENDPOINT)
  embeds both stored memories and search queries, enabling semantic retrieval.
  Namespaced per user_id. Used as an async context manager per-request.
  Store is passed to create_react_agent(store=store) AND through RunnableConfig
  so memory tools can access it without InjectedStore.

Short-term memory:
  AsyncCheckpointSaver — full graph state checkpointed to Postgres per thread_id
  after every node execution. Used as an async context manager per-request.
  Passed via create_react_agent(checkpointer=checkpointer).

Memory tools (get_user_memory, save_user_memory, delete_user_memory) are
returned by the memory_tools() factory and injected into the agent's tool list.
Tools access the store via RunnableConfig["configurable"]["store"], which the
streaming handler populates per-request.

Reference: https://github.com/databricks/app-templates/tree/main/agent-langgraph-long-term-memory
"""

import json
import logging
import os
from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks_langchain import AsyncCheckpointSaver, AsyncDatabricksStore
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from mlflow.types.responses import ResponsesAgentRequest


# ── Lakebase helpers ──────────────────────────────────────────────────────────

def _is_lakebase_hostname(value: str) -> bool:
    """Return True if value looks like a Lakebase connection hostname (not a short name)."""
    return ".database." in value and value.endswith(".com")


def resolve_lakebase_instance_name(
    instance_name: str, workspace_client: Optional[WorkspaceClient] = None
) -> str:
    """Resolve a Lakebase hostname to its short instance name if needed.

    When app.yaml uses valueFrom, Databricks injects the full connection hostname
    (e.g. instance-abc123.database.cloud.databricks.com). This function detects
    that case and resolves it to the short name via the SDK. If given a short
    name already (via value:), returns it unchanged.
    """
    if not _is_lakebase_hostname(instance_name):
        return instance_name

    client = workspace_client or WorkspaceClient()
    hostname = instance_name
    try:
        instances = list(client.database.list_database_instances())
    except Exception as exc:
        raise ValueError(
            f"Unable to list database instances to resolve hostname '{hostname}'. "
            "Ensure you have access to database instances."
        ) from exc

    for instance in instances:
        rw_dns = getattr(instance, "read_write_dns", None)
        ro_dns = getattr(instance, "read_only_dns", None)
        if hostname in (rw_dns, ro_dns):
            resolved = getattr(instance, "name", None)
            if not resolved:
                raise ValueError(
                    f"Found matching instance for hostname '{hostname}' "
                    "but instance name is not available."
                )
            logging.info(
                "Resolved Lakebase hostname '%s' to instance name '%s'", hostname, resolved
            )
            return resolved

    raise ValueError(
        f"Unable to find database instance matching hostname '{hostname}'. "
        "Ensure the hostname is correct and the instance exists."
    )


def get_user_id(request: ResponsesAgentRequest) -> Optional[str]:
    """Extract user_id from request custom_inputs or OBO context."""
    custom_inputs = dict(request.custom_inputs or {})
    if "user_id" in custom_inputs:
        return custom_inputs["user_id"]
    if request.context and getattr(request.context, "user_id", None):
        return request.context.user_id
    return None


def get_lakebase_access_error_message(instance_name: str) -> str:
    """Return a helpful error message when Lakebase access fails."""
    if os.getenv("DATABRICKS_APP_NAME"):
        app_name = os.getenv("DATABRICKS_APP_NAME")
        return (
            f"Failed to connect to Lakebase instance '{instance_name}'. "
            f"The App Service Principal for '{app_name}' may not have access.\n\n"
            "To fix this:\n"
            "1. Go to the Databricks UI and navigate to your app\n"
            "2. Click 'Edit' → 'App resources' → 'Add resource'\n"
            "3. Add your Lakebase instance with 'Can connect and create' permission\n"
            "4. Run setup_lakebase_permissions.py to grant schema and table permissions."
        )
    return (
        f"Failed to connect to Lakebase instance '{instance_name}'. "
        "Verify the instance name, your permissions, and Databricks authentication."
    )


# ── Store + checkpointer factories ───────────────────────────────────────────

def get_store(instance_name: str) -> AsyncDatabricksStore:
    """Return an AsyncDatabricksStore for the given Lakebase instance.

    Semantic search is powered by the configured embedding model
    (DATABRICKS_EMBEDDING_ENDPOINT). Use as an async context manager:

        async with get_store(instance_name) as store:
            await store.setup()
    """
    return AsyncDatabricksStore(
        instance_name=instance_name,
        embedding_endpoint=os.getenv(
            "DATABRICKS_EMBEDDING_ENDPOINT", "databricks-bge-large-en"
        ),
        embedding_dims=int(os.getenv("EMBEDDING_DIMS", "1024")),
    )


def get_checkpointer(instance_name: str) -> AsyncCheckpointSaver:
    """Return an AsyncCheckpointSaver for the given Lakebase instance.

    Use as an async context manager:

        async with get_checkpointer(instance_name) as checkpointer:
            await checkpointer.setup()
    """
    return AsyncCheckpointSaver(instance_name=instance_name)


# ── Memory tools — factory function ──────────────────────────────────────────
# Tools access the store via RunnableConfig["configurable"]["store"].
# The store is injected per-request by the streaming handler in server.py.

def memory_tools():
    """Return the long-term memory tool set: get, save, delete.

    Usage in agent init:
        tools = mcp_tools + [get_product_info] + memory_tools()
        config = {"configurable": {"user_id": user_id, "store": store, ...}}
    """

    @tool
    async def get_user_memory(query: str, config: RunnableConfig) -> str:
        """Search for relevant information about the user from long-term memory.

        Uses semantic search (pgvector + embedding model) so queries don't need
        to match stored text exactly. Call this at the start of every conversation
        to personalize responses based on what is known about the user.

        Args:
            query: What to search for (e.g. "user preferences", "past projects")
        """
        user_id = config.get("configurable", {}).get("user_id")
        if not user_id:
            return "Memory not available - no user_id provided."

        store: Optional[BaseStore] = config.get("configurable", {}).get("store")
        if not store:
            return "Memory not available - store not configured."

        namespace = ("user_memories", user_id.replace(".", "-"))
        results = await store.asearch(namespace, query=query, limit=5)

        if not results:
            return "No memories found for this user."

        memory_items = [f"- [{item.key}]: {json.dumps(item.value)}" for item in results]
        return f"Found {len(results)} relevant memories:\n" + "\n".join(memory_items)

    @tool
    async def save_user_memory(
        memory_key: str, memory_data_json: str, config: RunnableConfig
    ) -> str:
        """Save information about the user to long-term memory.

        Use this to remember user preferences, important details, or other
        information that should persist across conversations.

        Args:
            memory_key: A short descriptive key (e.g., "preferred_name", "team", "interests")
            memory_data_json: JSON object to save (e.g., '{"value": "engineering"}')
        """
        user_id = config.get("configurable", {}).get("user_id")
        if not user_id:
            return "Cannot save memory - no user_id provided."

        store: Optional[BaseStore] = config.get("configurable", {}).get("store")
        if not store:
            return "Cannot save memory - store not configured."

        namespace = ("user_memories", user_id.replace(".", "-"))
        try:
            memory_data = json.loads(memory_data_json)
            if not isinstance(memory_data, dict):
                return (
                    f"Failed: memory_data must be a JSON object, "
                    f"not {type(memory_data).__name__}"
                )
            await store.aput(namespace, memory_key, memory_data)
            return f"Successfully saved memory '{memory_key}' for user."
        except json.JSONDecodeError as e:
            return f"Failed to save memory: Invalid JSON - {e}"

    @tool
    async def delete_user_memory(memory_key: str, config: RunnableConfig) -> str:
        """Delete a specific memory from the user's long-term memory.

        Use this when the user asks to forget something or correct stored information.

        Args:
            memory_key: The key of the memory to delete (e.g., "preferred_name", "team")
        """
        user_id = config.get("configurable", {}).get("user_id")
        if not user_id:
            return "Cannot delete memory - no user_id provided."

        store: Optional[BaseStore] = config.get("configurable", {}).get("store")
        if not store:
            return "Cannot delete memory - store not configured."

        namespace = ("user_memories", user_id.replace(".", "-"))
        await store.adelete(namespace, memory_key)
        return f"Successfully deleted memory '{memory_key}' for user."

    return [get_user_memory, save_user_memory, delete_user_memory]
