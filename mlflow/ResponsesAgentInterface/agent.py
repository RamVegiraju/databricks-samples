import os
import mlflow
from typing import Annotated, Generator, TypedDict
from langchain_databricks import ChatDatabricks # The Databricks-specific provider
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from mlflow.models import set_model
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)


# --- 1. Define the Tool ---
@tool
def get_weather(location: str):
    """Returns the weather for a given location. Use this for weather queries."""
    # Production note: In Databricks, you might pull this from a Unity Catalog table
    return f"The weather in {location} is 18°C with light rain."

tools = [get_weather]
tool_node = ToolNode(tools)

# --- 2. Initialize the Model ---
# Using your specific FM model ID
llm = ChatDatabricks(
    endpoint="databricks-gpt-oss-120b",
    temperature=0.1 # Low temperature for reliable tool calling
).bind_tools(tools)

# --- 3. Define the LangGraph ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

def call_model(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def should_continue(state: State):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")
graph = workflow.compile()

# --- 4. The Agnostic MLflow Wrapper ---
class LangGraphResponsesAgent(ResponsesAgent):
    def __init__(self, agent):
        self.agent = agent

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])
        for _, events in self.agent.stream({"messages": cc_msgs}, stream_mode=["updates"]):
            for node_data in events.values():
                if "messages" in node_data:
                    yield from output_to_responses_items_stream(node_data["messages"])

# --- 5. Serving Setup ---
mlflow.langchain.autolog()
agent = LangGraphResponsesAgent(graph)
set_model(agent)