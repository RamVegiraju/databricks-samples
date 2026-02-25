import os
import json
import mlflow
from mlflow.entities import SpanType
from mlflow.genai.scorers import ToolCallCorrectness, scorer
from mlflow.entities import Feedback
from openai import OpenAI

# --- Setup ---
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks"))
mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
mlflow.openai.autolog()  # captures LLM spans (request/response, token usage)

client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=os.environ["DATABRICKS_BASE_URL"],
)

# --- Tool definition ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    }
]


# SpanType.TOOL is required so that ToolCallCorrectness can find this execution.
# Internally, mlflow.genai.evaluate runs extract_tools_called_from_trace() which
# does trace.search_spans(span_type=SpanType.TOOL) — spans without an explicit
# TOOL type are invisible to the scorer, so actual_calls would always be empty
# and every row would be scored as incorrect.
@mlflow.trace(span_type=SpanType.TOOL)
def get_weather(city: str) -> str:
    """Stubbed weather lookup."""
    db = {
        "San Francisco": "65°F, Foggy",
        "New York": "72°F, Sunny",
        "Seattle": "55°F, Rainy",
    }
    return db.get(city, "Weather data unavailable")


# --- Agent (tool-calling loop) ---
@mlflow.trace
def run_agent(query: str) -> str:
    messages = [{"role": "user", "content": query}]

    while True:
        resp = client.chat.completions.create(
            model="databricks-claude-sonnet-4-6",
            messages=messages,
            tools=TOOLS,
            max_tokens=256,
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = get_weather(**args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            return msg.content


# --- Predict wrapper for mlflow.genai.evaluate ---
# MLflow 3.x unpacks `inputs` as kwargs, so the param name must match the key.
def predict(query: str) -> str:
    return run_agent(query)


# --- BYO scorer (demo) ---
@scorer
def is_not_empty(outputs: str) -> Feedback:
    ok = len(outputs.strip()) > 10
    return Feedback(
        value=ok,
        rationale="Response has content." if ok else "Response looks empty or trivial.",
    )


# --- Evaluation dataset ---
# expectations.expected_tool_calls is used by ToolCallCorrectness for fuzzy
# matching (default should_exact_match=False).  Each entry needs "name" and
# optionally "arguments"; omitting "arguments" on any call skips argument
# checking for all calls in that row.
eval_data = [
    {
        "inputs": {"query": "What's the weather in San Francisco?"},
        "expectations": {
            "expected_tool_calls": [
                {"name": "get_weather", "arguments": {"city": "San Francisco"}},
            ]
        },
    },
    {
        "inputs": {"query": "What's the weather in New York?"},
        "expectations": {
            "expected_tool_calls": [
                {"name": "get_weather", "arguments": {"city": "New York"}},
            ]
        },
    },
    {
        "inputs": {"query": "What's the weather in Seattle?"},
        "expectations": {
            "expected_tool_calls": [
                {"name": "get_weather", "arguments": {"city": "Seattle"}},
            ]
        },
    },
]

if __name__ == "__main__":
    results = mlflow.genai.evaluate(
        predict_fn=predict,
        data=eval_data,
        scorers=[ToolCallCorrectness(), is_not_empty],
    )
    print(results.metrics)
