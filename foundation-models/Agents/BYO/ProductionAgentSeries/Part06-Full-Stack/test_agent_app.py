"""
test_agent_app.py — End-to-end validation of the deployed Part06 agent app.

Tests the full tool surface across non-streaming and streaming modes:
  - get_product_info  (inline Python tool)
  - add              (MCP remote tool)
  - return_biodata   (MCP remote tool)
  - memory           (save + retrieve across two turns)

Requirements:
    pip install requests python-dotenv

Usage:
    # Against deployed app
    python test_agent_app.py --app-url https://<agent-app>.databricksapps.com

    # Against local server (python start_server.py)
    python test_agent_app.py --app-url http://localhost:8000
"""

import argparse
import json
import os
import sys
import uuid
from typing import Any

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == THIS_DIR:
    sys.path.pop(0)

import requests
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(THIS_DIR, "agent_app/.env"))


def _headers(app_url: str) -> dict:
    # Local dev doesn't need auth
    if "localhost" in app_url or "127.0.0.1" in app_url:
        return {"Content-Type": "application/json"}
    # Deployed Databricks Apps require OAuth — use SDK to get a valid token
    token = WorkspaceClient(profile="oauth").config.authenticate()["Authorization"].split(" ")[1]
    return {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}


def invoke(app_url: str, prompt: str, thread_id: str, user_id: str) -> dict[str, Any]:
    payload = {
        "input": [{"role": "user", "content": prompt}],
        "custom_inputs": {"thread_id": thread_id, "user_id": user_id},
    }
    resp = requests.post(
        f"{app_url.rstrip('/')}/invocations",
        json=payload,
        headers=_headers(app_url),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def invoke_stream(app_url: str, prompt: str, thread_id: str, user_id: str) -> str:
    payload = {
        "input": [{"role": "user", "content": prompt}],
        "stream": True,
        "custom_inputs": {"thread_id": thread_id, "user_id": user_id},
    }
    resp = requests.post(
        f"{app_url.rstrip('/')}/invocations",
        json=payload,
        headers=_headers(app_url),
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    text_parts: list[str] = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        raw = line[len("data: "):]
        if raw == "[DONE]":
            break
        try:
            evt = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if evt.get("type") == "response.output_text.delta":
            text_parts.append(evt.get("delta", ""))
    return "".join(text_parts)


def extract_text(data: dict[str, Any]) -> str:
    for item in data.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return content.get("text", "")
    return ""


def run_test(label: str, app_url: str, prompt: str, thread_id: str, user_id: str, stream: bool = False) -> None:
    mode = "stream" if stream else "non-stream"
    print(f"\n{'='*60}")
    print(f"[{label}] ({mode})")
    print(f"Prompt : {prompt}")
    if stream:
        text = invoke_stream(app_url, prompt, thread_id, user_id)
        print(f"Reply  : {text}")
    else:
        data = invoke(app_url, prompt, thread_id, user_id)
        text = extract_text(data)
        tool_calls = [i for i in data.get("output", []) if i.get("type") == "function_call"]
        print(f"Tools  : {[t.get('name') for t in tool_calls]}")
        print(f"Reply  : {text}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-url", required=True, help="Deployed agent app base URL")
    parser.add_argument(
        "--mode",
        choices=["non-stream", "stream", "both"],
        default="both",
        help="Run tests in non-streaming mode, streaming mode, or both (default: both)",
    )
    args = parser.parse_args()

    thread_id = f"test-{uuid.uuid4().hex[:8]}"
    user_id = "test-user"

    print(f"App URL   : {args.app_url}")
    print(f"Thread ID : {thread_id}")
    print(f"User ID   : {user_id}")
    print(f"Mode      : {args.mode}")

    run_non_stream = args.mode in ("non-stream", "both")
    run_stream = args.mode in ("stream", "both")

    # ── Non-streaming pass ────────────────────────────────────────────────────
    if run_non_stream:
        print(f"\n{'='*60}")
        print("NON-STREAMING PASS")

        run_test("get_product_info", args.app_url, "What is MLflow?", thread_id, user_id)
        run_test("MCP add", args.app_url, "Add 7 and 8.", thread_id, user_id)
        run_test("MCP biodata", args.app_url, "Look up the biodata for Alice.", thread_id, user_id)

        mem_thread = f"mem-{uuid.uuid4().hex[:8]}"
        run_test("memory save", args.app_url, "My name is Alice and I love hiking.", mem_thread, user_id)
        run_test("memory retrieve", args.app_url, "What do you know about me?", mem_thread, user_id)

    # ── Streaming pass ────────────────────────────────────────────────────────
    if run_stream:
        print(f"\n{'='*60}")
        print("STREAMING PASS")

        stream_thread = f"stream-{uuid.uuid4().hex[:8]}"
        run_test("get_product_info", args.app_url, "What is MLflow?", stream_thread, user_id, stream=True)
        run_test("MCP add", args.app_url, "Add 7 and 8.", stream_thread, user_id, stream=True)
        run_test("MCP biodata", args.app_url, "Look up the biodata for Alice.", stream_thread, user_id, stream=True)

        mem_stream_thread = f"mem-stream-{uuid.uuid4().hex[:8]}"
        run_test("memory save", args.app_url, "My name is Bob and I love cycling.", mem_stream_thread, user_id, stream=True)
        run_test("memory retrieve", args.app_url, "What do you know about me?", mem_stream_thread, user_id, stream=True)

    print(f"\n{'='*60}")
    print("All tests complete.")


if __name__ == "__main__":
    main()
