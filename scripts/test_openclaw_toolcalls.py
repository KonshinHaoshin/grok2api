#!/usr/bin/env python3
"""
Minimal OpenClaw-style tool-calling smoke test for grok2api.

Checks:
1) assistant returns tool_calls with finish_reason=tool_calls
2) tool result can be sent back with role=tool
3) model continues and returns final assistant text
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


def _post_json(url: str, payload: Dict[str, Any], api_key: str, timeout: int) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Request failed: {e}") from e


def _extract_choice(resp: Dict[str, Any]) -> Dict[str, Any]:
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Invalid response (missing choices): {resp}")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError(f"Invalid response choice: {choice}")
    return choice


def _extract_tool_calls(choice: Dict[str, Any]) -> List[Dict[str, Any]]:
    msg = choice.get("message") or {}
    tool_calls = msg.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [tc for tc in tool_calls if isinstance(tc, dict)]


def _first_tool_call_name(tool_calls: List[Dict[str, Any]]) -> Optional[str]:
    if not tool_calls:
        return None
    fn = tool_calls[0].get("function") or {}
    if not isinstance(fn, dict):
        return None
    name = fn.get("name")
    return name if isinstance(name, str) else None


def main() -> int:
    parser = argparse.ArgumentParser(description="grok2api OpenClaw tool-calling smoke test")
    parser.add_argument("--base-url", default=os.getenv("GROK2API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.getenv("GROK2API_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("GROK2API_MODEL", "grok-4"))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("GROK2API_TIMEOUT", "60")))
    args = parser.parse_args()

    if not args.api_key:
        print("FAIL: missing api key. Set --api-key or GROK2API_API_KEY.", file=sys.stderr)
        return 2

    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    tool_def = {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file to workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    }

    step1_payload = {
        "model": args.model,
        "stream": False,
        "tools": [tool_def],
        "tool_choice": "required",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use tool write_file to create file hello_openclaw.txt with content "
                    "'hello from tool call'. Only call the tool."
                ),
            }
        ],
    }

    print("STEP 1: requesting tool call...")
    step1 = _post_json(endpoint, step1_payload, args.api_key, args.timeout)
    choice1 = _extract_choice(step1)
    finish1 = choice1.get("finish_reason")
    tool_calls = _extract_tool_calls(choice1)
    if finish1 != "tool_calls":
        print(f"FAIL: expected finish_reason=tool_calls, got {finish1!r}", file=sys.stderr)
        return 1
    if not tool_calls:
        print("FAIL: no tool_calls in step1 response", file=sys.stderr)
        return 1
    first_name = _first_tool_call_name(tool_calls)
    if first_name != "write_file":
        print(f"FAIL: expected tool name 'write_file', got {first_name!r}", file=sys.stderr)
        return 1
    print(f"PASS: got {len(tool_calls)} tool_call(s), first={first_name}")

    first_call = tool_calls[0]
    first_call_id = first_call.get("id")
    if not isinstance(first_call_id, str) or not first_call_id:
        print("FAIL: missing tool_call id in step1 response", file=sys.stderr)
        return 1

    step2_messages: List[Dict[str, Any]] = [
        step1_payload["messages"][0],
        {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        },
        {
            "role": "tool",
            "tool_call_id": first_call_id,
            "name": "write_file",
            "content": json.dumps(
                {"ok": True, "path": "hello_openclaw.txt", "bytes": 20},
                ensure_ascii=False,
            ),
        },
        {
            "role": "user",
            "content": "Tool finished. Confirm completion in one short sentence.",
        },
    ]

    step2_payload = {
        "model": args.model,
        "stream": False,
        "tools": [tool_def],
        "tool_choice": "auto",
        "messages": step2_messages,
    }

    print("STEP 2: sending tool result and requesting final answer...")
    step2 = _post_json(endpoint, step2_payload, args.api_key, args.timeout)
    choice2 = _extract_choice(step2)
    finish2 = choice2.get("finish_reason")
    msg2 = choice2.get("message") or {}
    content2 = msg2.get("content")
    if finish2 not in ("stop", "length"):
        print(f"FAIL: expected finish_reason stop/length, got {finish2!r}", file=sys.stderr)
        return 1
    if not isinstance(content2, str) or not content2.strip():
        print(f"FAIL: final assistant content is empty: {content2!r}", file=sys.stderr)
        return 1

    print("PASS: final response received")
    print("--- Final assistant message ---")
    print(content2.strip())
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

