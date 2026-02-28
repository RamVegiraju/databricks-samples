"""
memory.py - AI operations for the memory system.

Two responsibilities only:
  - embed(texts)        embed a list of texts via bge-large-en (AI Gateway)
  - summarize(messages) ask Claude to summarize a session

No database logic here. All reads/writes go through db.py.
"""

import os
from openai import OpenAI


def embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts via the Databricks bge-large-en endpoint on AI Gateway.
    Returns a list of 1024-dim float vectors.
    """
    client = OpenAI(
        api_key=os.environ["DATABRICKS_TOKEN"],
        base_url=os.environ["DATABRICKS_BASE_URL"],
    )
    response = client.embeddings.create(
        input=texts,
        model=os.environ["DATABRICKS_EMBEDDING_ENDPOINT"],
    )
    return [item.embedding for item in response.data]


def summarize(messages: list[dict], client: OpenAI) -> str:
    """
    Ask Claude to produce a concise summary of a completed session.
    messages: list of {"role": ..., "content": ...} dicts.
    Returns the summary as a plain string.
    """
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )
    response = client.chat.completions.create(
        model="databricks-gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Summarize the following conversation in 3-5 concise sentences. "
                    "Cover: topics discussed, decisions made, and where things left off. "
                    "Write in third person (e.g. 'The user asked about...')."
                ),
            },
            {"role": "user", "content": transcript},
        ],
        max_tokens=300,
    )
    raw = response.choices[0].message.content
    if isinstance(raw, list):
        return "\n".join(block["text"] for block in raw if block.get("type") == "text")
    return raw
