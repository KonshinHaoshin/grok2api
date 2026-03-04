#!/usr/bin/env python3
"""
tool_call.py 内部工具函数单元测试。
不依赖运行中的服务，直接 import 模块执行。

覆盖范围：
  _balance_braces     — 修复 1：数组截断补全
  parse_tool_call_block — 修复 2：arguments=None → "{}"
  parse_tool_calls    — 端到端：多 block、无效 block、mode 参数
  format_tool_history — 历史转换
  _repair_json        — 组合修复管道

用法：
  uv run python scripts/test_tool_call_utils.py
"""

from __future__ import annotations

import json
import sys
import os
from typing import Any, Callable

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# 直接导入，然后替换模块内的 get_config 引用
# ---------------------------------------------------------------------------

import app.services.grok.utils.tool_call as _tc_module

_CONFIG_DEFAULTS = {
    "compat.tool_call_mode": "strict",
}

def _get_config_stub(key: str, default: Any = None) -> Any:
    return _CONFIG_DEFAULTS.get(key, default)

_tc_module.get_config = _get_config_stub  # type: ignore[attr-defined]

from app.services.grok.utils.tool_call import (  # noqa: E402
    _balance_braces,
    _repair_json,
    format_tool_history,
    parse_tool_call_block,
    parse_tool_calls,
)


# ---------------------------------------------------------------------------
# 测试框架（零依赖）
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def case(name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        try:
            fn()
            _results.append((name, True, ""))
        except AssertionError as e:
            _results.append((name, False, str(e)))
        except Exception as e:
            _results.append((name, False, f"{type(e).__name__}: {e}"))
        return fn
    return decorator


def eq(actual: Any, expected: Any, msg: str = "") -> None:
    assert actual == expected, (
        f"{msg + ': ' if msg else ''}expected {expected!r}, got {actual!r}"
    )


def contains(haystack: str, needle: str, msg: str = "") -> None:
    assert needle in haystack, (
        f"{msg + ': ' if msg else ''}expected {needle!r} in {haystack!r}"
    )


def not_none(val: Any, msg: str = "") -> None:
    assert val is not None, f"{msg + ': ' if msg else ''}expected non-None"


def is_none(val: Any, msg: str = "") -> None:
    assert val is None, f"{msg + ': ' if msg else ''}expected None, got {val!r}"


# ---------------------------------------------------------------------------
# _balance_braces 测试（修复 1）
# ---------------------------------------------------------------------------

@case("balance_braces: 完整 JSON 不变")
def _():
    eq(_balance_braces('{"a": 1}'), '{"a": 1}')


@case("balance_braces: 缺少闭合 } 补全")
def _():
    result = _balance_braces('{"a": 1')
    eq(result, '{"a": 1}')


@case("balance_braces: 缺少闭合 ] 补全（修复1核心）")
def _():
    result = _balance_braces('{"files": ["a.txt", "b.txt"')
    assert result.endswith("]}"), f"expected ending ']}}', got {result!r}"
    json.loads(result)   # 补全后必须可解析


@case("balance_braces: 嵌套数组+对象同时截断")
def _():
    # {"list": [{"k": "v"  ← 缺 }]}
    result = _balance_braces('{"list": [{"k": "v"')
    json.loads(result)   # 能解析即通过


@case("balance_braces: 字符串内的括号不计入")
def _():
    # 字符串里有 { [ 不应影响计数
    result = _balance_braces('{"msg": "hello {world} [ok]"}')
    eq(result, '{"msg": "hello {world} [ok]"}')


@case("balance_braces: 空字符串返回空")
def _():
    eq(_balance_braces(""), "")


# ---------------------------------------------------------------------------
# _repair_json 测试
# ---------------------------------------------------------------------------

@case("repair_json: 截断数组参数可修复")
def _():
    raw = '{"name": "upload", "arguments": {"files": ["a.txt", "b.txt"'
    result = _repair_json(raw)
    not_none(result, "repair_json 应返回非 None")
    eq(result["name"], "upload")
    assert isinstance(result["arguments"]["files"], list)


@case("repair_json: 尾随逗号")
def _():
    result = _repair_json('{"a": 1,}')
    not_none(result)
    eq(result["a"], 1)


@case("repair_json: code fence 包裹")
def _():
    result = _repair_json('```json\n{"x": 42}\n```')
    not_none(result)
    eq(result["x"], 42)


@case("repair_json: 无法修复时返回 None")
def _():
    is_none(_repair_json("not json at all %%%"))


# ---------------------------------------------------------------------------
# parse_tool_call_block 测试（修复 2：arguments=None）
# ---------------------------------------------------------------------------

@case("parse_tool_call_block: 正常 arguments 对象")
def _():
    raw = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
    result = parse_tool_call_block(raw)
    not_none(result)
    eq(result["function"]["name"], "get_weather")
    args = json.loads(result["function"]["arguments"])
    eq(args["city"], "Beijing")


@case("parse_tool_call_block: arguments=null → '{}'（修复2核心）")
def _():
    raw = '{"name": "ping", "arguments": null}'
    result = parse_tool_call_block(raw)
    not_none(result, "arguments=null 不应被丢弃")
    eq(result["function"]["arguments"], "{}")


@case("parse_tool_call_block: arguments 缺失 → '{}'")
def _():
    raw = '{"name": "ping"}'
    result = parse_tool_call_block(raw)
    not_none(result)
    eq(result["function"]["arguments"], "{}")


@case("parse_tool_call_block: arguments 是数组")
def _():
    raw = '{"name": "batch", "arguments": [1, 2, 3]}'
    result = parse_tool_call_block(raw)
    not_none(result)
    args = json.loads(result["function"]["arguments"])
    eq(args, [1, 2, 3])


@case("parse_tool_call_block: 截断数组 arguments + strict 模式下修复后可解析")
def _():
    # 模拟流式截断：arguments 数组不完整
    raw = '{"name": "upload", "arguments": {"files": ["a.txt", "b.txt"'
    result = parse_tool_call_block(raw, mode="strict")
    not_none(result, "修复管道应能补全截断数组")
    args = json.loads(result["function"]["arguments"])
    assert isinstance(args["files"], list)


@case("parse_tool_call_block: 不在 tools 列表中 → None")
def _():
    tools = [{"type": "function", "function": {"name": "allowed"}}]
    raw = '{"name": "not_allowed", "arguments": {}}'
    is_none(parse_tool_call_block(raw, tools=tools))


@case("parse_tool_call_block: compatible 模式保留不可解析 arguments")
def _():
    raw = '{"name": "fn", "arguments": "not valid json %%"}'
    is_none(parse_tool_call_block(raw, mode="strict"))
    result = parse_tool_call_block(raw, mode="compatible")
    not_none(result, "compatible 模式应保留 tool call")


@case("parse_tool_call_block: name 缺失 → None")
def _():
    raw = '{"arguments": {"x": 1}}'
    is_none(parse_tool_call_block(raw))


# ---------------------------------------------------------------------------
# parse_tool_calls 端到端测试
# ---------------------------------------------------------------------------

@case("parse_tool_calls: 单个 block")
def _():
    content = 'Sure!\n<tool_call>{"name": "get_weather", "arguments": {"city": "Shanghai"}}</tool_call>'
    text, tcs = parse_tool_calls(content)
    not_none(tcs)
    eq(len(tcs), 1)
    eq(tcs[0]["function"]["name"], "get_weather")


@case("parse_tool_calls: 多个 block")
def _():
    content = (
        '<tool_call>{"name": "fn_a", "arguments": {"x": 1}}</tool_call>\n'
        '<tool_call>{"name": "fn_b", "arguments": {"y": 2}}</tool_call>'
    )
    text, tcs = parse_tool_calls(content)
    not_none(tcs)
    eq(len(tcs), 2)
    eq(tcs[0]["function"]["name"], "fn_a")
    eq(tcs[1]["function"]["name"], "fn_b")


@case("parse_tool_calls: 无 block → 返回原始内容")
def _():
    content = "Just a normal reply."
    text, tcs = parse_tool_calls(content)
    eq(text, content)
    is_none(tcs)


@case("parse_tool_calls: 无效 block 保留在 text")
def _():
    content = '<tool_call>INVALID%%%</tool_call> some text'
    text, tcs = parse_tool_calls(content)
    is_none(tcs)
    contains(text, "<tool_call>")


@case("parse_tool_calls: 部分有效部分无效 block")
def _():
    content = (
        '<tool_call>{"name": "ok", "arguments": {}}</tool_call>'
        '<tool_call>INVALID</tool_call>'
    )
    text, tcs = parse_tool_calls(content)
    not_none(tcs)
    eq(len(tcs), 1)
    # 无效 block 保留在 text
    not_none(text)
    contains(text, "<tool_call>INVALID</tool_call>")


# ---------------------------------------------------------------------------
# format_tool_history 测试
# ---------------------------------------------------------------------------

@case("format_tool_history: assistant tool_calls → text")
def _():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
                }
            ],
        }
    ]
    result = format_tool_history(messages)
    eq(len(result), 1)
    eq(result[0]["role"], "assistant")
    contains(result[0]["content"], "<tool_call>")
    contains(result[0]["content"], "get_weather")


@case("format_tool_history: tool role → user text")
def _():
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_abc",
            "name": "get_weather",
            "content": '{"temp": "18°C"}',
        }
    ]
    result = format_tool_history(messages)
    eq(len(result), 1)
    eq(result[0]["role"], "user")
    contains(result[0]["content"], "<tool_result>")
    contains(result[0]["content"], "call_abc")


@case("format_tool_history: assistant tool_calls arguments=None → 不崩溃")
def _():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_x",
                    "type": "function",
                    "function": {"name": "ping", "arguments": None},
                }
            ],
        }
    ]
    result = format_tool_history(messages)
    eq(len(result), 1)
    # arguments 应为空对象，而非 null
    payload = json.loads(
        result[0]["content"].split("<tool_call>")[1].split("</tool_call>")[0]
    )
    eq(payload["arguments"], {})


@case("format_tool_history: 普通消息原样保留")
def _():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    result = format_tool_history(messages)
    eq(result, messages)


# ---------------------------------------------------------------------------
# 输出结果
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"\n{'═' * 60}")
    print(f"  tool_call.py 单元测试  ({len(_results)} cases)")
    print(f"{'═' * 60}")

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = len(_results) - passed

    for name, ok, msg in _results:
        status = "PASS" if ok else "FAIL"
        line = f"  {status}  {name}"
        if not ok:
            line += f"\n        └─ {msg}"
        print(line)

    print(f"{'─' * 60}")
    print(f"  {passed} passed, {failed} failed")
    if failed == 0:
        print("  ALL TESTS PASSED")
    print(f"{'═' * 60}\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
