import argparse
import json
from typing import Any

import requests


def invoke(base_url: str, user_prompt: str) -> dict[str, Any]:
    payload = {"input": [{"role": "user", "content": user_prompt}]}
    response = requests.post(
        f"{base_url.rstrip('/')}/invocations",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def summarize_response(data: dict[str, Any]) -> None:
    output = data.get("output", [])
    has_tool_call = any(item.get("type") == "function_call" for item in output)
    has_tool_result = any(item.get("type") == "function_call_output" for item in output)
    assistant_text = None
    for item in output:
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    assistant_text = content.get("text")
                    break

    print(f"tool_call_detected: {has_tool_call}")
    print(f"tool_output_detected: {has_tool_result}")
    if assistant_text:
        print(f"assistant_text: {assistant_text}")
    print("raw_response:")
    print(json.dumps(data, indent=2))


def run_example(base_url: str, mode: str) -> None:
    if mode == "no-tool":
        prompt = "Give me one short sentence about mountains."
    elif mode == "tool":
        prompt = "What is the weather in Seattle?"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"\n=== Running mode: {mode} ===")
    print(f"prompt: {prompt}")
    result = invoke(base_url, prompt)
    summarize_response(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Invoke local ResponsesAgent endpoint.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL for mlflow models serve endpoint.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "no-tool", "tool"],
        default="all",
        help="Run no-tool, tool, or both examples.",
    )
    args = parser.parse_args()

    if args.mode == "all":
        run_example(args.base_url, "no-tool")
        run_example(args.base_url, "tool")
    else:
        run_example(args.base_url, args.mode)


if __name__ == "__main__":
    main()
