#!/usr/bin/env python3
"""
验证 Bug 1~4 修复的测试脚本。

测试项：
  T1  单轮工具调用 (非流式) — finish_reason=tool_calls，返回 tool_calls 字段
  T2  多轮工具对话，最后一轮 tools=None — 模型仍能看到历史上下文，返回正常文本  (Bug 1)
  T3  普通对话不受工具提示词污染 — 返回正常文本，不含 <tool_call> 标签         (Bug 2)
  T4  流式模式工具调用 — 最终 chunk finish_reason=tool_calls                   (Bug 4)
  T5  (可选) 图片模型带 tools 参数 — 不报错，正常返回图片内容                   (Bug 3)

用法：
  uv run python scripts/test_tool_call_bugfixes.py \\
      --base-url http://127.0.0.1:8000 \\
      --api-key YOUR_KEY \\
      --model grok-3

环境变量（可替代命令行参数）：
  GROK2API_BASE_URL   GROK2API_API_KEY   GROK2API_MODEL   GROK2API_TIMEOUT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

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


def _post_stream(url: str, payload: Dict[str, Any], api_key: str, timeout: int) -> List[Dict[str, Any]]:
    """返回所有非 [DONE] SSE chunk 的解析结果。"""
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
    chunks: List[Dict[str, Any]] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunks.append(json.loads(data))
                except json.JSONDecodeError:
                    pass
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Request failed: {e}") from e
    return chunks


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def _choice(resp: Dict[str, Any]) -> Dict[str, Any]:
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AssertionError(f"No choices in response: {resp}")
    return choices[0]


def _tool_calls(choice: Dict[str, Any]) -> List[Dict[str, Any]]:
    msg = choice.get("message") or {}
    tcs = msg.get("tool_calls")
    return tcs if isinstance(tcs, list) else []


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    },
}


def test_t1_single_turn(endpoint: str, model: str, api_key: str, timeout: int) -> Tuple[bool, str, Optional[List]]:
    """T1: 单轮工具调用，非流式，tool_choice=required。"""
    payload = {
        "model": model,
        "stream": False,
        "tools": [TOOL_DEF],
        "tool_choice": "required",
        "messages": [
            {"role": "user", "content": "What's the weather in Beijing? Use the tool."},
        ],
    }
    resp = _post_json(endpoint, payload, api_key, timeout)
    choice = _choice(resp)
    finish = choice.get("finish_reason")
    tcs = _tool_calls(choice)
    if finish != "tool_calls":
        return False, f"finish_reason={finish!r}, want 'tool_calls'", None
    if not tcs:
        return False, "tool_calls list is empty", None
    fn = tcs[0].get("function", {})
    if fn.get("name") != "get_weather":
        return False, f"tool name={fn.get('name')!r}, want 'get_weather'", None
    return True, f"finish_reason=tool_calls, {len(tcs)} call(s), name=get_weather", tcs


def test_t2_multi_turn_no_tools(
    endpoint: str, model: str, api_key: str, timeout: int, tool_calls_from_t1: List[Dict]
) -> Tuple[bool, str]:
    """T2 (Bug 1): 多轮工具对话，最后一轮不传 tools。

    如果 Bug 1 未修复，模型看不到之前的工具调用历史，
    可能输出混乱或拒绝回答。
    """
    if not tool_calls_from_t1:
        return False, "skipped: no tool_calls from T1"

    first_id = tool_calls_from_t1[0].get("id", "call_001")
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": "What's the weather in Beijing? Use the tool."},
        {"role": "assistant", "content": None, "tool_calls": tool_calls_from_t1},
        {
            "role": "tool",
            "tool_call_id": first_id,
            "name": "get_weather",
            "content": json.dumps({"city": "Beijing", "temp": "18°C", "condition": "Sunny"}),
        },
        {"role": "user", "content": "Thanks. Give me a one-sentence summary of the weather."},
    ]
    payload = {
        "model": model,
        "stream": False,
        # Bug 1 关键：最后一轮不传 tools，仍需正确识别历史
        "messages": messages,
    }
    resp = _post_json(endpoint, payload, api_key, timeout)
    choice = _choice(resp)
    finish = choice.get("finish_reason")
    msg = choice.get("message") or {}
    content = msg.get("content", "")
    if finish not in ("stop", "length"):
        return False, f"finish_reason={finish!r}, want stop/length"
    if not isinstance(content, str) or not content.strip():
        return False, f"content is empty: {content!r}"
    return True, f"finish={finish}, content={content.strip()[:80]!r}"


def test_t3_no_pollution(endpoint: str, model: str, api_key: str, timeout: int) -> Tuple[bool, str]:
    """T3 (Bug 2): 普通对话不传 tools，回复不应含 <tool_call> 标签。"""
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "user", "content": "Say 'hello world' and nothing else."},
        ],
    }
    resp = _post_json(endpoint, payload, api_key, timeout)
    choice = _choice(resp)
    msg = choice.get("message") or {}
    content = msg.get("content", "") or ""
    finish = choice.get("finish_reason")
    if "<tool_call>" in content.lower():
        return False, f"Response contains <tool_call> tag: {content[:200]!r}"
    if finish not in ("stop", "length"):
        return False, f"finish_reason={finish!r}, want stop/length"
    if not content.strip():
        return False, "content is empty"
    return True, f"clean response: {content.strip()[:80]!r}"


def test_t4_stream_tool_calls(endpoint: str, model: str, api_key: str, timeout: int) -> Tuple[bool, str]:
    """T4 (Bug 4): 流式模式工具调用，最终 chunk 应有 finish_reason=tool_calls。"""
    payload = {
        "model": model,
        "stream": True,
        "tools": [TOOL_DEF],
        "tool_choice": "required",
        "messages": [
            {"role": "user", "content": "What's the weather in Shanghai? Use the tool."},
        ],
    }
    chunks = _post_stream(endpoint, payload, api_key, timeout)
    if not chunks:
        return False, "no SSE chunks received"

    # 找到包含 finish_reason 的 chunk
    finish_chunks = [
        c for c in chunks
        if isinstance(c.get("choices"), list)
        and c["choices"]
        and c["choices"][0].get("finish_reason") is not None
    ]
    if not finish_chunks:
        return False, f"no chunk with finish_reason among {len(chunks)} chunks"

    last_finish = finish_chunks[-1]["choices"][0]["finish_reason"]
    if last_finish != "tool_calls":
        return False, f"final finish_reason={last_finish!r}, want 'tool_calls'"

    # 确认某个 chunk 包含 tool_calls delta
    tool_chunks = [
        c for c in chunks
        if isinstance(c.get("choices"), list)
        and c["choices"]
        and c["choices"][0].get("delta", {}).get("tool_calls")
    ]
    if not tool_chunks:
        return False, "no chunk contains tool_calls delta"

    return True, f"got {len(tool_chunks)} tool_calls chunk(s), final finish_reason=tool_calls"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _run(label: str, fn, *args) -> bool:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    try:
        result = fn(*args)
        if isinstance(result, tuple) and len(result) >= 2:
            ok, msg = result[0], result[1]
            extra = result[2] if len(result) > 2 else None
        else:
            ok, msg, extra = bool(result), "", None
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {msg}")
        return ok, extra
    except Exception as e:
        print(f"  ERROR: {e}")
        return False, None


def main() -> int:
    parser = argparse.ArgumentParser(description="grok2api tool-call bugfix verification")
    parser.add_argument("--base-url", default=os.getenv("GROK2API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.getenv("GROK2API_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("GROK2API_MODEL", "grok-3"))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("GROK2API_TIMEOUT", "60")))
    parser.add_argument("--skip-stream", action="store_true", help="Skip T4 streaming test")
    args = parser.parse_args()

    if not args.api_key:
        print("FAIL: missing api key. Set --api-key or GROK2API_API_KEY.", file=sys.stderr)
        return 2

    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    m = args.model
    k = args.api_key
    t = args.timeout

    results = []

    ok1, tcs = _run("T1: 单轮工具调用 (非流式, tool_choice=required)", test_t1_single_turn, endpoint, m, k, t)
    results.append(("T1 单轮工具调用", ok1))

    ok2, _ = _run("T2: 多轮工具对话, 最后一轮不传 tools  [Bug 1]", test_t2_multi_turn_no_tools, endpoint, m, k, t, tcs or [])
    results.append(("T2 多轮历史(Bug1)", ok2))

    ok3, _ = _run("T3: 普通对话无工具污染  [Bug 2]", test_t3_no_pollution, endpoint, m, k, t)
    results.append(("T3 无污染(Bug2)", ok3))

    if not args.skip_stream:
        ok4, _ = _run("T4: 流式工具调用 finish_reason  [Bug 4]", test_t4_stream_tool_calls, endpoint, m, k, t)
        results.append(("T4 流式工具(Bug4)", ok4))

    print(f"\n{'═'*60}")
    print("  SUMMARY")
    print(f"{'═'*60}")
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\n  ALL TESTS PASSED")
        return 0
    else:
        print("\n  SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
