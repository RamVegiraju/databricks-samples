import argparse
import json
import os
import sys
from typing import Any

# This file is intentionally named requests.py.
# Remove this directory from import lookup so Python imports the third-party
# `requests` package (not this file itself).
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == THIS_DIR:
    sys.path.pop(0)

import requests


def invoke_non_stream(base_url: str, prompt: str) -> dict[str, Any]:
    payload = {"input": [{"role": "user", "content": prompt}]}
    resp = requests.post(
        f"{base_url.rstrip('/')}/invocations",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def invoke_stream(base_url: str, prompt: str) -> tuple[list[dict[str, Any]], str]:
    payload = {"input": [{"role": "user", "content": prompt}], "stream": True}
    resp = requests.post(
        f"{base_url.rstrip('/')}/invocations",
        json=payload,
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    events: list[dict[str, Any]] = []
    text_parts: list[str] = []

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        raw = line[len("data: ") :]
        if raw == "[DONE]":
            break
        try:
            evt = json.loads(raw)
        except json.JSONDecodeError:
            continue
        events.append(evt)
        if evt.get("type") == "response.output_text.delta":
            text_parts.append(evt.get("delta", ""))

    return events, "".join(text_parts)


def summarize_non_stream(data: dict[str, Any]) -> None:
    output = data.get("output", [])
    has_tool_call = any(item.get("type") == "function_call" for item in output)
    has_tool_output = any(item.get("type") == "function_call_output" for item in output)
    assistant_text = ""
    for item in output:
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    assistant_text = content.get("text", "")
                    break

    print("non-stream summary")
    print(f"  tool_call_detected: {has_tool_call}")
    print(f"  tool_output_detected: {has_tool_output}")
    print(f"  assistant_text: {assistant_text}")
    print("  full_response:")
    print(json.dumps(data, indent=2))


def summarize_stream(events: list[dict[str, Any]], streamed_text: str) -> None:
    tool_calls = 0
    tool_outputs = 0
    done_messages = 0
    delta_events = 0
    for evt in events:
        evt_type = evt.get("type")
        if evt_type == "response.output_text.delta":
            delta_events += 1
        elif evt_type == "response.output_item.done":
            item = evt.get("item", {})
            item_type = item.get("type")
            if item_type == "function_call":
                tool_calls += 1
            elif item_type == "function_call_output":
                tool_outputs += 1
            elif item_type == "message":
                done_messages += 1

    print("stream summary")
    print(f"  tool_call_events: {tool_calls}")
    print(f"  tool_output_events: {tool_outputs}")
    print(f"  text_delta_events: {delta_events}")
    print(f"  final_message_events: {done_messages}")
    print(f"  reconstructed_stream_text: {streamed_text}")
    print("  stream_events:")
    print(json.dumps(events, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Invoke AgentServing in non-stream and stream modes.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt", default="What is the weather in Seattle?")
    args = parser.parse_args()

    print(f"base_url: {args.base_url}")
    print(f"prompt: {args.prompt}")
    print()

    print("=== non-stream invoke ===")
    non_stream = invoke_non_stream(args.base_url, args.prompt)
    summarize_non_stream(non_stream)
    print()

    print("=== stream invoke ===")
    events, text = invoke_stream(args.base_url, args.prompt)
    summarize_stream(events, text)


if __name__ == "__main__":
    main()
