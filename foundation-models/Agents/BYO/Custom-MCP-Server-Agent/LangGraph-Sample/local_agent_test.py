import asyncio
from typing import Annotated, Any, Optional, Sequence, TypedDict

from databricks.sdk import WorkspaceClient
from databricks_langchain import (
    ChatDatabricks,
    DatabricksMCPServer,
    DatabricksMultiServerMCPClient,
)

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode


# ----------------------------
# Config
# ----------------------------
LLM_ENDPOINT_NAME = "databricks-gpt-oss-120b"  # change if needed
CUSTOM_MCP_APP_URL = "https://custom-mcp-server-7474655512364318.aws.databricksapps.com/mcp"
DBX_PROFILE = "oauth"  # your Databricks CLI profile

SYSTEM_PROMPT = """
You are a helpful assistant.

Tool use policy:
- Only call tools when the user explicitly asks for a tool result, or when a tool is necessary to answer correctly.
- For addition, even if you have general knowledge, use the add tool to compute the result.
- If the question can be answered from general knowledge, answer directly WITHOUT calling tools.
- If you do call tools, use the minimum number of tool calls needed.
- After tool use, incorporate the tool result into your final answer.
""".strip()


# ----------------------------
# LangGraph state
# ----------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]


def summarize_tool_usage(messages: Sequence[AnyMessage]):
    """Return (tool_calls_requested, tool_results_returned)."""
    tool_calls = []
    tool_returns = []

    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            tool_calls.extend(m.tool_calls)

        if isinstance(m, ToolMessage):
            tool_returns.append(
                {
                    "tool_name": m.name,
                    "tool_call_id": m.tool_call_id,
                    "content": m.content,
                }
            )

    return tool_calls, tool_returns


def create_tool_calling_graph(model, tools, system_prompt: Optional[str] = None):
    # Bind tools to the model (enables tool calling)
    model = model.bind_tools(tools)

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    # Prepend system prompt if provided
    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}] + list(state["messages"])
        )
    else:
        preprocessor = RunnableLambda(lambda state: list(state["messages"]))

    model_runnable = preprocessor | model

    def call_model(state: AgentState):
        resp = model_runnable.invoke(state)
        return {"messages": [resp]}

    g = StateGraph(AgentState)
    g.add_node("agent", RunnableLambda(call_model))
    g.add_node("tools", ToolNode(tools))

    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")

    return g.compile()


async def run_query(graph, text: str):
    out = await graph.ainvoke({"messages": [HumanMessage(content=text)]})
    final_text = out["messages"][-1].content

    tool_calls, tool_returns = summarize_tool_usage(out["messages"])

    print("\n==============================")
    print("USER:", text)
    print("ASSISTANT:", final_text)
    print("TOOL_CALLS_REQUESTED:", tool_calls)
    print("TOOL_RESULTS_RETURNED:", tool_returns)
    print("==============================\n")

    return out


async def main():
    # Local auth: uses your Databricks CLI oauth profile
    ws = WorkspaceClient(profile=DBX_PROFILE)

    # LLM from Databricks FM API
    llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

    # MCP server hosted on Databricks Apps
    mcp_client = DatabricksMultiServerMCPClient(
        [
            DatabricksMCPServer(
                name="custom_mcp",
                url=CUSTOM_MCP_APP_URL,
                workspace_client=ws,
            )
        ]
    )

    # IMPORTANT: get_tools() is async
    mcp_tools = await mcp_client.get_tools()
    print("Loaded MCP tools:", [t.name for t in mcp_tools])

    graph = create_tool_calling_graph(llm, mcp_tools, SYSTEM_PROMPT)

    # Should use tool
    await run_query(graph, "Add 2 and 3.")

    # Should use tool
    await run_query(graph, "Return biodata for name Ram.")

    # Should NOT use tool (general knowledge)
    await run_query(graph, "What is the capital of France? Return only the city name.")


if __name__ == "__main__":
    asyncio.run(main())
