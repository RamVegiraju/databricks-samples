"""
memory.py - AI operations for the memory system.

Two responsibilities only:
  - embed(texts)        embed a list of texts via bge-embed-ram
  - summarize(messages) ask Claude to summarize a session

No database logic here. All reads/writes go through db.py.
"""

import os
import json
import urllib.request
import urllib.error
from typing import Optional
from openai import OpenAI


def embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts via the Databricks bge-embed-ram serving endpoint.
    Returns a list of 1024-dim float vectors.
    Handles both OpenAI-style {"data":[{"embedding":[...]}]}
    and Databricks-style {"embeddings":[[...]]} response shapes.
    """
    host     = os.environ["DATABRICKS_HOST"].rstrip("/")
    endpoint = os.environ["DATABRICKS_EMBEDDING_ENDPOINT"]
    token    = os.environ["DATABRICKS_TOKEN"]
    url      = f"{host}/serving-endpoints/{endpoint}/invocations"

    body = json.dumps({"input": texts}).encode("utf-8")
    req  = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            out = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode() if e.fp else str(e)
        raise RuntimeError(f"Embedding API error {e.code}: {err_body}") from e

    if "data" in out and isinstance(out["data"], list):
        return [item["embedding"] for item in out["data"]]
    if "embeddings" in out:
        return out["embeddings"]
    raise ValueError(f"Unexpected embedding response shape: {list(out.keys())}")


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
        model="databricks-claude-opus-4-6",
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
    return response.choices[0].message.content
