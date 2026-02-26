"""
evaluate.py — MLflow GenAI evaluation for the Part06 agent.

Evaluates the inline get_product_info tool using the same pattern as Part05:
  - mlflow.openai.autolog() captures LLM spans automatically
  - @mlflow.trace(span_type=SpanType.TOOL) makes tool spans visible to scorers
  - @mlflow.trace on run_agent owns the top-level trace
  - ToolCallCorrectness (built-in LLM judge) checks name + argument correctness
  - response_mentions_product (custom @scorer) validates the final response

Note: This evaluation runs standalone against the Databricks FM API — it does
not require the agent_app server to be running.  The tool definitions here
mirror agent_app/agent/tools.py so eval results map directly to production.

Run:
    cd eval/
    python evaluate.py
"""

import json
import os

import mlflow
from mlflow.entities import Feedback, SpanType
from mlflow.genai.scorers import ToolCallCorrectness, scorer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../agent_app/.env"))

# ── MLflow setup ──────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks"))
mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
mlflow.openai.autolog()  # captures LLM spans (request/response, token usage)

client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=f"{os.environ['DATABRICKS_HOST'].rstrip('/')}/serving-endpoints",
)

# ── Tool definition (mirrors agent_app/agent/tools.py) ───────────────────────
# SpanType.TOOL is required: ToolCallCorrectness calls
# trace.search_spans(span_type=SpanType.TOOL) internally.
# Without this annotation, actual_calls is always empty and every row scores False.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_product_info",
            "description": "Returns product details from the Databricks catalog for a given product name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Name of the product"},
                },
                "required": ["product_name"],
            },
        },
    }
]

_CATALOG = {
    "Databricks Lakehouse": (
        "Unified data + AI platform combining data lake and warehouse capabilities "
        "with ACID transactions, schema enforcement, and native ML integration."
    ),
    "MLflow": (
        "Open-source ML lifecycle management platform covering experiment tracking, "
        "projects, model packaging, registry, and GenAI evaluation."
    ),
    "Delta Lake": (
        "ACID-compliant open-source storage layer over data lakes, supporting schema "
        "enforcement, time travel, and scalable metadata handling."
    ),
    "Apache Spark": (
        "Unified analytics engine for large-scale data processing with modules for "
        "SQL, streaming, machine learning, and graph processing."
    ),
}


@mlflow.trace(span_type=SpanType.TOOL)
def get_product_info(product_name: str) -> str:
    """Stubbed product lookup — mirrors the production tool in tools.py."""
    info = _CATALOG.get(product_name)
    if info:
        return info
    available = ", ".join(_CATALOG.keys())
    return f"Product '{product_name}' not found. Available: {available}"


# ── Agent (tool-calling loop) ─────────────────────────────────────────────────

@mlflow.trace
def run_agent(query: str) -> str:
    messages = [{"role": "user", "content": query}]

    while True:
        resp = client.chat.completions.create(
            model="databricks-claude-sonnet-4-6",
            messages=messages,
            tools=TOOLS,
            max_tokens=512,
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = get_product_info(**args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            return msg.content


# predict_fn for mlflow.genai.evaluate — param name must match eval_data inputs key
def predict(query: str) -> str:
    return run_agent(query)


# ── Custom scorer ─────────────────────────────────────────────────────────────

@scorer
def response_is_informative(outputs: str) -> Feedback:
    """Checks that the response is substantive (not an error or empty reply)."""
    ok = len(outputs.strip()) > 30 and "not found" not in outputs.lower()
    return Feedback(
        value=ok,
        rationale=(
            "Response contains substantive product information."
            if ok
            else "Response is too short or indicates the product was not found."
        ),
    )


# ── Evaluation dataset ────────────────────────────────────────────────────────
# expected_tool_calls format: all entries must include "arguments" for per-argument
# scoring.  Omitting "arguments" on any entry disables argument checking for that row.

eval_data = [
    {
        "inputs": {"query": "Tell me about Databricks Lakehouse"},
        "expectations": {
            "expected_tool_calls": [
                {"name": "get_product_info", "arguments": {"product_name": "Databricks Lakehouse"}},
            ]
        },
    },
    {
        "inputs": {"query": "What is MLflow?"},
        "expectations": {
            "expected_tool_calls": [
                {"name": "get_product_info", "arguments": {"product_name": "MLflow"}},
            ]
        },
    },
    {
        "inputs": {"query": "Describe Delta Lake"},
        "expectations": {
            "expected_tool_calls": [
                {"name": "get_product_info", "arguments": {"product_name": "Delta Lake"}},
            ]
        },
    },
    {
        "inputs": {"query": "What can you tell me about Apache Spark?"},
        "expectations": {
            "expected_tool_calls": [
                {"name": "get_product_info", "arguments": {"product_name": "Apache Spark"}},
            ]
        },
    },
]


if __name__ == "__main__":
    results = mlflow.genai.evaluate(
        predict_fn=predict,
        data=eval_data,
        scorers=[ToolCallCorrectness(), response_is_informative],
    )
    print(results.metrics)
