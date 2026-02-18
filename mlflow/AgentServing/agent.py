import json
import logging
import os
from typing import Any, AsyncGenerator, AsyncIterator
from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import tool
from langchain_databricks import ChatDatabricks
from fastapi import HTTPException
from langgraph.prebuilt import create_react_agent
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    create_text_delta,
    output_to_responses_items_stream,
    to_chat_completions_input,
)

logger = logging.getLogger(__name__)

LLM_ENDPOINT = os.getenv("AGENT_LLM_ENDPOINT", "databricks-gpt-oss-120b")
LLM_TEMPERATURE = float(os.getenv("AGENT_LLM_TEMPERATURE", "0.1"))


@tool
def get_weather(location: str):
    """Returns the weather for a given location. Use this for weather queries."""
    return f"The weather in {location} is 18°C with light rain."


tools = [get_weather]
llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=LLM_TEMPERATURE)
agent = create_react_agent(model=llm, tools=tools)


def _chunk_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                text_parts.append(str(item["text"]))
        return "".join(text_parts)
    return ""


def _normalize_ai_message_content(msg: Any) -> None:
    # MLflow's stream validator expects message content text fields to be strings.
    # Databricks model chunks can include mixed list payloads (reasoning + text dicts).
    if isinstance(msg, AIMessage) and not isinstance(msg.content, str):
        msg.content = _chunk_text_content(msg.content)


async def process_agent_astream_events(
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
                    # Some providers emit cumulative text; convert to true delta if so.
                    previous = text_seen_by_item.get(item_id, "")
                    if content.startswith(previous):
                        delta = content[len(previous) :]
                        text_seen_by_item[item_id] = content
                    else:
                        delta = content
                        text_seen_by_item[item_id] = previous + content

                    if not delta:
                        continue
                    yield ResponsesAgentStreamEvent(
                        **create_text_delta(delta=delta, item_id=item_id)
                    )


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
    messages = {"messages": to_chat_completions_input([item.model_dump() for item in request.input])}
    try:
        async for event in process_agent_astream_events(
            agent.astream(input=messages, stream_mode=["updates", "messages"])
        ):
            yield event
    except Exception as exc:
        logger.exception("Agent streaming failed")
        raise HTTPException(status_code=500, detail=f"Streaming failed: {exc}") from exc

