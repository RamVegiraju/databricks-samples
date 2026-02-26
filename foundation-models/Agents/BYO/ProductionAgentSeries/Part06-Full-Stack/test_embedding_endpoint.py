"""
test_embedding_endpoint.py — Validate the databricks-bge-large-en managed endpoint.

This is the embedding endpoint used by DatabricksStore for semantic memory search.
No provisioning needed — databricks-bge-large-en is a pay-per-token Foundation Model
API endpoint managed by Databricks.

Requirements:
    pip install openai python-dotenv

Usage:
    python test_embedding_endpoint.py
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "agent_app/.env"))

ENDPOINT = os.getenv("DATABRICKS_EMBEDDING_ENDPOINT", "databricks-bge-large-en")

client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=f"{os.environ['DATABRICKS_HOST'].rstrip('/')}/serving-endpoints",
)


def main():
    print(f"Testing embedding endpoint: {ENDPOINT}\n")

    texts = [
        "Databricks is a unified data and AI platform.",
        "LangGraph is a framework for building stateful agents.",
    ]

    response = client.embeddings.create(model=ENDPOINT, input=texts)

    for i, item in enumerate(response.data):
        embedding = item.embedding
        print(f"Input  : {texts[i]}")
        print(f"Dims   : {len(embedding)}")
        print(f"Sample : {embedding[:5]}")
        print()

    print("Embedding endpoint OK.")


if __name__ == "__main__":
    main()
