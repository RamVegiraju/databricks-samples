"""
server.py — AgentServer endpoint definitions using the ResponsesAgent interface.

Registers two endpoints via MLflow AgentServer decorators:
  @invoke()  — non-streaming, collects all output items
  @stream()  — streaming, yields ResponsesAgentStreamEvent via SSE

Both handlers:
  - open AsyncDatabricksStore + AsyncCheckpointSaver as async context managers
    per-request (correct connection pool lifecycle)
  - call init_agent() with the open store + checkpointer
  - pass store, user_id, thread_id through RunnableConfig so memory tools and
    the checkpointer can scope state correctly per user and session
  - handle Lakebase access errors with helpful messages

Invoke payload example:
  {
    "input": [{"role": "user", "content": "What is MLflow?"}],
    "custom_inputs": {"thread_id": "session-abc", "user_id": "alice"}
  }
"""

import json
import logging
from typing import Any, AsyncGenerator, AsyncIterator
from uuid import uuid4

from fastapi import HTTPException
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    create_text_delta,
    output_to_responses_items_stream,
    to_chat_completions_input,
)

from .agent import LAKEBASE_INSTANCE_NAME, init_agent, sp_workspace_client
from .memory import (
    get_checkpointer,
    get_lakebase_access_error_message,
    get_store,
    get_user_id,
)

logger = logging.getLogger(__name__)


# ── Stream event helpers (unchanged from Part03) ──────────────────────────────

def _chunk_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "".join(parts)
    return ""


def _normalize_ai_message_content(msg: Any) -> None:
    # MLflow's stream validator requires message content to be a string.
    # Some Databricks models return mixed list payloads (reasoning + text dicts).
    if isinstance(msg, AIMessage) and not isinstance(msg.content, str):
        msg.content = _chunk_text_content(msg.content)


async def _process_agent_astream_events(
    async_stream: AsyncIterator[Any],
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    text_seen_by_item: dict[str, str] = {}

    async for event in async_stream:
        if len(event) == 2:
            mode, payload = event
        else:
            _, mode, payload = event

        if mode == "updates":
            for node_data in payload.values():
                messages = node_data.get("messages", [])
                if not messages:
                    continue
                for msg in messages:
                    _normalize_ai_message_content(msg)
                    if isinstance(msg, ToolMessage) and not isinstance(msg.content, str):
                        msg.content = json.dumps(msg.content)
                for item in output_to_responses_items_stream(iter(messages)):
                    yield item

        elif mode == "messages":
            chunk = payload[0]
            if isinstance(chunk, AIMessageChunk):
                item_id = chunk.id or str(uuid4())
                content = _chunk_text_content(chunk.content)
                if content:
                    previous = text_seen_by_item.get(item_id, "")
                    if content.startswith(previous):
                        delta = content[len(previous):]
                        text_seen_by_item[item_id] = content
                    else:
                        delta = content
                        text_seen_by_item[item_id] = previous + content
                    if not delta:
                        continue
                    yield ResponsesAgentStreamEvent(
                        **create_text_delta(delta=delta, item_id=item_id)
                    )


# ── AgentServer endpoints ─────────────────────────────────────────────────────

@invoke()
async def non_streaming(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    outputs = [
        event.item
        async for event in streaming(request)
        if event.type == "response.output_item.done"
    ]
    return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)


@stream()
async def streaming(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    user_id = get_user_id(request)
    thread_id = (request.custom_inputs or {}).get("thread_id", str(uuid4()))

    messages: dict[str, Any] = {
        "messages": to_chat_completions_input([item.model_dump() for item in request.input])
    }

    try:
        async with get_store(LAKEBASE_INSTANCE_NAME) as store:
            await store.setup()
            async with get_checkpointer(LAKEBASE_INSTANCE_NAME) as checkpointer:
                await checkpointer.setup()

                # thread_id → CheckpointSaver (short-term, within-session)
                # user_id   → memory tools via store (long-term, cross-session)
                # store     → memory tools access it via RunnableConfig
                config: dict[str, Any] = {
                    "configurable": {
                        "thread_id": thread_id,
                        "store": store,
                    }
                }
                if user_id:
                    config["configurable"]["user_id"] = user_id

                agent = await init_agent(
                    store=store,
                    checkpointer=checkpointer,
                    workspace_client=sp_workspace_client,
                )

                async for event in _process_agent_astream_events(
                    agent.astream(
                        input=messages,
                        config=config,
                        stream_mode=["updates", "messages"],
                    )
                ):
                    yield event

    except Exception as exc:
        error_msg = str(exc).lower()
        if any(k in error_msg for k in ["permission", "lakebase", "pool", "connect"]):
            logger.error("Lakebase access error: %s", exc)
            raise HTTPException(
                status_code=503,
                detail=get_lakebase_access_error_message(LAKEBASE_INSTANCE_NAME),
            ) from exc
        logger.exception("Agent streaming failed")
        raise HTTPException(status_code=500, detail=f"Streaming failed: {exc}") from exc
