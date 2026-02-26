from langchain_core.tools import tool

# Product catalog — the inline Python tool bundled directly with the agent.
# This demonstrates a plain function tool alongside MCP-sourced tools,
# showing both patterns living together in the same agent.
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


@tool
def get_product_info(product_name: str) -> str:
    """Returns product details from the Databricks catalog for a given product name."""
    info = _CATALOG.get(product_name)
    if info:
        return info
    available = ", ".join(_CATALOG.keys())
    return f"Product '{product_name}' not found. Available products: {available}"
