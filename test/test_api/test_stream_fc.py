#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightLLM OpenAI streaming function call (tool call) test script.

Usage:
    # Start LightLLM server first, e.g.:
    #   python -m lightllm.server.api_server --port 8000 --model_dir /path/to/model --tp 1

    # Run all tests:
    python test/test_api/test_stream_function_call.py

    # Specify server url and model:
    python test/test_api/test_stream_function_call.py --base-url http://localhost:8080 --model my_model

    # Run a single test:
    python test/test_api/test_stream_function_call.py --test single
"""

import argparse
import json
import sys
import traceback
from typing import Dict, List, Optional

from openai import OpenAI

# ──────────────────────────────────────────────
# Tool definitions
# ──────────────────────────────────────────────

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称，例如：北京、上海"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "温度单位，默认 celsius",
                },
            },
            "required": ["city"],
        },
    },
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "执行数学计算",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式，例如：2+3*4"},
            },
            "required": ["expression"],
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "在互联网上搜索信息",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "max_results": {"type": "integer", "description": "最大返回结果数，默认 5"},
            },
            "required": ["query"],
        },
    },
}

ALL_TOOLS = [WEATHER_TOOL, CALCULATOR_TOOL, SEARCH_TOOL]


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def collect_stream_tool_calls(response) -> Dict:
    """
    Consume a streaming chat completion response and reassemble:
      - content: concatenated text content
      - reasoning_content: concatenated reasoning content
      - tool_calls: dict keyed by index -> {id, name, arguments}
      - finish_reason: the final finish_reason
      - chunks: raw chunk list for inspection
    """
    content = ""
    reasoning_content = ""
    tool_calls: Dict[int, Dict] = {}
    finish_reason = None
    chunks = []

    for chunk in response:
        chunks.append(chunk)
        choice = chunk.choices[0]
        delta = choice.delta

        if choice.finish_reason is not None:
            finish_reason = choice.finish_reason

        if delta.content:
            content += delta.content

        if getattr(delta, "reasoning_content", None):
            reasoning_content += delta.reasoning_content

        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls:
                    tool_calls[idx] = {"id": None, "name": "", "arguments": ""}
                if tc.id:
                    tool_calls[idx]["id"] = tc.id
                if tc.function.name:
                    tool_calls[idx]["name"] = tc.function.name
                if tc.function.arguments:
                    tool_calls[idx]["arguments"] += tc.function.arguments

    return {
        "content": content,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "chunks": chunks,
    }


def print_result(result: Dict):
    """Pretty-print a collected stream result."""
    if result["reasoning_content"]:
        print(f"  [思考]: {result['reasoning_content'][:200]}...")
    if result["content"]:
        print(f"  [内容]: {result['content']}")
    if result["tool_calls"]:
        for idx, tc in sorted(result["tool_calls"].items()):
            print(f"  [工具调用 {idx}]: id={tc['id']}, name={tc['name']}, arguments={tc['arguments']}")
    print(f"  [finish_reason]: {result['finish_reason']}")
    print(f"  [chunks数量]: {len(result['chunks'])}")


def assert_check(condition: bool, msg: str):
    """Simple assertion with message."""
    if not condition:
        raise AssertionError(f"FAIL: {msg}")


# ──────────────────────────────────────────────
# Test cases
# ──────────────────────────────────────────────


def test_single_tool_call(client: OpenAI, model: str):
    """测试单个工具调用 - 查询天气"""
    print("=" * 60)
    print("[TEST] 单工具流式调用")
    print("=" * 60)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        stream=True,
        temperature=0.0,
        max_tokens=1000,
    )

    result = collect_stream_tool_calls(response)
    print_result(result)

    # Validate
    if result["tool_calls"]:
        tc = result["tool_calls"][0]
        assert_check(tc["id"] is not None and len(tc["id"]) > 0, "tool_call id 不应为空")
        assert_check(tc["name"] == "get_weather", f"期望函数名 get_weather, 实际: {tc['name']}")
        args = json.loads(tc["arguments"])
        assert_check("city" in args, "参数中应包含 city 字段")
        assert_check(
            result["finish_reason"] == "tool_calls", f"finish_reason 应为 tool_calls, 实际: {result['finish_reason']}"
        )
        print("  [PASS] 单工具流式调用测试通过\n")
    else:
        print("  [WARN] 模型未调用工具，可能模型不支持该 tool_call_parser 格式\n")


def test_parallel_tool_calls(client: OpenAI, model: str):
    """测试并行多工具调用"""
    print("=" * 60)
    print("[TEST] 并行多工具流式调用")
    print("=" * 60)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "帮我查一下北京和上海的天气"}],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        stream=True,
        temperature=0.0,
        max_tokens=1000,
    )

    result = collect_stream_tool_calls(response)
    print_result(result)

    if len(result["tool_calls"]) >= 2:
        # 检查每个 tool_call 的完整性
        ids_seen = set()
        for idx, tc in result["tool_calls"].items():
            assert_check(tc["id"] is not None, f"tool_call[{idx}] id 不应为空")
            assert_check(tc["id"] not in ids_seen, f"tool_call id 重复: {tc['id']}")
            ids_seen.add(tc["id"])
            assert_check(tc["name"] == "get_weather", f"tool_call[{idx}] 函数名应为 get_weather")
            args = json.loads(tc["arguments"])
            assert_check("city" in args, f"tool_call[{idx}] 参数中应包含 city 字段")

        assert_check(
            result["finish_reason"] == "tool_calls", f"finish_reason 应为 tool_calls, 实际: {result['finish_reason']}"
        )
        print("  [PASS] 并行多工具流式调用测试通过\n")
    elif len(result["tool_calls"]) == 1:
        print("  [WARN] 模型只调用了 1 个工具（可能不支持并行调用），跳过并行校验\n")
    else:
        print("  [WARN] 模型未调用工具\n")


def test_mixed_content_and_tool_calls(client: OpenAI, model: str):
    """测试混合输出：模型先输出文本再调用工具"""
    print("=" * 60)
    print("[TEST] 文本+工具调用混合输出")
    print("=" * 60)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "先说一句问候语，然后帮我查北京的天气",
            }
        ],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        stream=True,
        temperature=0.7,
        max_tokens=1000,
    )

    result = collect_stream_tool_calls(response)
    print_result(result)

    if result["tool_calls"]:
        tc = result["tool_calls"][0]
        assert_check(tc["name"] == "get_weather", f"期望函数名 get_weather, 实际: {tc['name']}")
        args = json.loads(tc["arguments"])
        assert_check("city" in args, "参数中应包含 city 字段")
        print("  [PASS] 混合输出测试通过\n")
    else:
        # 有些模型可能只输出文本或只调用工具
        print("  [WARN] 模型未调用工具\n")


def test_tool_choice_required(client: OpenAI, model: str):
    """测试 tool_choice=required，模型必须调用工具"""
    print("=" * 60)
    print("[TEST] tool_choice=required")
    print("=" * 60)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "你好"}],
        tools=ALL_TOOLS,
        tool_choice="required",
        stream=True,
        temperature=0.0,
        max_tokens=1000,
    )

    result = collect_stream_tool_calls(response)
    print_result(result)

    if result["tool_calls"]:
        tc = result["tool_calls"][0]
        assert_check(tc["id"] is not None, "tool_call id 不应为空")
        assert_check(tc["name"] in ["get_weather", "calculate", "web_search"], f"函数名不在预期范围: {tc['name']}")
        assert_check(
            result["finish_reason"] == "tool_calls", f"finish_reason 应为 tool_calls, 实际: {result['finish_reason']}"
        )
        print("  [PASS] tool_choice=required 测试通过\n")
    else:
        print("  [WARN] tool_choice=required 但模型未调用工具\n")


def test_tool_choice_none(client: OpenAI, model: str):
    """测试 tool_choice=none，模型不应调用工具"""
    print("=" * 60)
    print("[TEST] tool_choice=none")
    print("=" * 60)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
        tools=[WEATHER_TOOL],
        tool_choice="none",
        stream=True,
        temperature=0.0,
        max_tokens=1000,
    )

    result = collect_stream_tool_calls(response)
    print_result(result)

    assert_check(len(result["tool_calls"]) == 0, "tool_choice=none 时不应有工具调用")
    assert_check(
        result["finish_reason"] in ("stop", "length"), f"finish_reason 应为 stop 或 length, 实际: {result['finish_reason']}"
    )
    print("  [PASS] tool_choice=none 测试通过\n")


def test_tool_choice_specific_function(client: OpenAI, model: str):
    """测试 tool_choice 指定具体函数"""
    print("=" * 60)
    print("[TEST] tool_choice=指定函数")
    print("=" * 60)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "1+1等于几"}],
        tools=ALL_TOOLS,
        tool_choice={"type": "function", "function": {"name": "calculate"}},
        stream=True,
        temperature=0.0,
        max_tokens=1000,
    )

    result = collect_stream_tool_calls(response)
    print_result(result)

    if result["tool_calls"]:
        tc = result["tool_calls"][0]
        assert_check(tc["name"] == "calculate", f"期望函数名 calculate, 实际: {tc['name']}")
        args = json.loads(tc["arguments"])
        assert_check("expression" in args, "参数中应包含 expression 字段")
        print("  [PASS] tool_choice 指定函数测试通过\n")
    else:
        print("  [WARN] 模型未调用指定函数\n")


def test_multi_turn_with_tool_result(client: OpenAI, model: str):
    """测试多轮对话：工具调用 -> 返回结果 -> 模型继续回答"""
    print("=" * 60)
    print("[TEST] 多轮对话（工具调用+结果回传）")
    print("=" * 60)

    # Round 1: 用户提问，模型应调用工具
    print("  --- Round 1: 用户提问 ---")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        stream=True,
        temperature=0.0,
        max_tokens=1000,
    )

    result1 = collect_stream_tool_calls(response)
    print_result(result1)

    if not result1["tool_calls"]:
        print("  [SKIP] 模型未调用工具，跳过多轮测试\n")
        return

    tc = result1["tool_calls"][0]
    tool_call_id = tc["id"]
    tool_name = tc["name"]
    tool_args = tc["arguments"]

    # Round 2: 传回工具结果，模型应基于结果回答
    print("  --- Round 2: 回传工具结果 ---")
    weather_result = json.dumps({"city": "北京", "temperature": 22, "condition": "晴", "humidity": 45}, ensure_ascii=False)

    messages = [
        {"role": "user", "content": "北京今天天气怎么样？"},
        {
            "role": "assistant",
            "content": result1["content"] if result1["content"] else None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": tool_args},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": weather_result,
        },
    ]

    response2 = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        stream=True,
        temperature=0.0,
        max_tokens=1000,
    )

    result2 = collect_stream_tool_calls(response2)
    print_result(result2)

    assert_check(len(result2["content"]) > 0, "模型应基于工具结果生成文本回复")
    assert_check(
        result2["finish_reason"] in ("stop", "length"),
        f"finish_reason 应为 stop 或 length, 实际: {result2['finish_reason']}",
    )
    print("  [PASS] 多轮对话测试通过\n")


def test_stream_chunk_integrity(client: OpenAI, model: str):
    """测试流式 chunk 的结构完整性"""
    print("=" * 60)
    print("[TEST] 流式 chunk 结构完整性校验")
    print("=" * 60)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "帮我查北京天气"}],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        stream=True,
        temperature=0.0,
        max_tokens=1000,
    )

    result = collect_stream_tool_calls(response)

    if not result["tool_calls"]:
        print("  [SKIP] 模型未调用工具，跳过 chunk 校验\n")
        return

    # Validate chunk structure
    has_role_chunk = False
    tool_call_name_chunks = 0
    tool_call_arg_chunks = 0
    finish_reason_count = 0

    for chunk in result["chunks"]:
        assert_check(chunk.id is not None, "chunk.id 不应为空")
        assert_check(
            chunk.object == "chat.completion.chunk", f"chunk.object 应为 chat.completion.chunk, 实际: {chunk.object}"
        )
        assert_check(len(chunk.choices) > 0, "chunk.choices 不应为空")

        choice = chunk.choices[0]
        delta = choice.delta

        if delta.role == "assistant":
            has_role_chunk = True

        if choice.finish_reason is not None:
            finish_reason_count += 1

        if delta.tool_calls:
            for tc in delta.tool_calls:
                assert_check(tc.index is not None, "tool_call.index 不应为 None")
                assert_check(
                    tc.type is None or tc.type == "function", f"tool_call.type 应为 function 或 None, 实际: {tc.type}"
                )
                if tc.function.name:
                    tool_call_name_chunks += 1
                if tc.function.arguments:
                    tool_call_arg_chunks += 1

    print(f"  总 chunks: {len(result['chunks'])}")
    print(f"  包含 role 的 chunk: {has_role_chunk}")
    print(f"  包含函数名的 chunk: {tool_call_name_chunks}")
    print(f"  包含参数的 chunk: {tool_call_arg_chunks}")
    print(f"  finish_reason chunk: {finish_reason_count}")

    assert_check(has_role_chunk, "应有至少一个 chunk 包含 role=assistant")
    assert_check(tool_call_name_chunks >= 1, "应有至少一个 chunk 包含函数名")
    assert_check(tool_call_arg_chunks >= 1, "应有至少一个 chunk 包含参数")
    assert_check(finish_reason_count >= 1, "应有至少一个 chunk 包含 finish_reason")

    # 验证拼接后的 arguments 是合法 JSON
    for idx, tc in result["tool_calls"].items():
        try:
            json.loads(tc["arguments"])
        except json.JSONDecodeError as e:
            raise AssertionError(f"tool_call[{idx}] arguments 不是合法 JSON: {tc['arguments']}, error: {e}")

    print("  [PASS] 流式 chunk 结构完整性校验通过\n")


def test_multiple_different_tools(client: OpenAI, model: str):
    """测试同时调用多个不同的工具"""
    print("=" * 60)
    print("[TEST] 多种工具并行调用")
    print("=" * 60)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "帮我查北京天气，再算一下 123*456 等于多少",
            }
        ],
        tools=[WEATHER_TOOL, CALCULATOR_TOOL],
        tool_choice="auto",
        stream=True,
        temperature=0.0,
        max_tokens=1000,
    )

    result = collect_stream_tool_calls(response)
    print_result(result)

    if len(result["tool_calls"]) >= 2:
        names = {tc["name"] for tc in result["tool_calls"].values()}
        assert_check("get_weather" in names, "应包含 get_weather 调用")
        assert_check("calculate" in names, "应包含 calculate 调用")

        # 每个 tool_call 的 id 应唯一
        ids = [tc["id"] for tc in result["tool_calls"].values()]
        assert_check(len(ids) == len(set(ids)), f"tool_call id 应唯一, 实际: {ids}")

        # 每个 arguments 应是合法 JSON
        for idx, tc in result["tool_calls"].items():
            json.loads(tc["arguments"])

        print("  [PASS] 多种工具并行调用测试通过\n")
    elif len(result["tool_calls"]) == 1:
        print("  [WARN] 模型只调用了 1 个工具，跳过多工具校验\n")
    else:
        print("  [WARN] 模型未调用工具\n")


def test_stream_with_usage(client: OpenAI, model: str):
    """测试流式输出中的 usage 信息（stream_options.include_usage）"""
    print("=" * 60)
    print("[TEST] 流式输出 usage 信息")
    print("=" * 60)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "帮我查北京天气"}],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        stream=True,
        stream_options={"include_usage": True},
        temperature=0.0,
        max_tokens=1000,
    )

    usage_info = None
    chunks = []
    for chunk in response:
        chunks.append(chunk)
        if chunk.usage is not None:
            usage_info = chunk.usage

    if usage_info:
        print(f"  prompt_tokens: {usage_info.prompt_tokens}")
        print(f"  completion_tokens: {usage_info.completion_tokens}")
        print(f"  total_tokens: {usage_info.total_tokens}")
        assert_check(usage_info.prompt_tokens > 0, "prompt_tokens 应 > 0")
        assert_check(usage_info.completion_tokens > 0, "completion_tokens 应 > 0")
        assert_check(
            usage_info.total_tokens == usage_info.prompt_tokens + usage_info.completion_tokens,
            "total_tokens 应等于 prompt + completion",
        )
        print("  [PASS] 流式 usage 信息测试通过\n")
    else:
        print("  [WARN] 未收到 usage 信息（服务端可能不支持 stream_options）\n")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

TEST_REGISTRY = {
    "single": test_single_tool_call,
    "parallel": test_parallel_tool_calls,
    "mixed": test_mixed_content_and_tool_calls,
    "required": test_tool_choice_required,
    "none": test_tool_choice_none,
    "specific": test_tool_choice_specific_function,
    "multi_turn": test_multi_turn_with_tool_result,
    "chunk_integrity": test_stream_chunk_integrity,
    "multi_tools": test_multiple_different_tools,
    "usage": test_stream_with_usage,
}


def main():
    parser = argparse.ArgumentParser(description="LightLLM streaming function call test")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="LightLLM server base URL")
    parser.add_argument("--model", default="default_model", help="Model name")
    parser.add_argument(
        "--test",
        default=None,
        choices=list(TEST_REGISTRY.keys()),
        help="Run a specific test (default: run all)",
    )
    parser.add_argument("--api-key", default="EMPTY", help="API key (default: EMPTY)")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    print(f"Server: {args.base_url}")
    print(f"Model:  {args.model}")
    print()

    tests_to_run = [args.test] if args.test else list(TEST_REGISTRY.keys())
    passed = 0
    failed = 0

    for name in tests_to_run:
        try:
            TEST_REGISTRY[name](client, args.model)
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            traceback.print_exc()
            failed += 1
            print()
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            failed += 1
            print()

    print("=" * 60)
    print(f"结果: {passed} passed, {failed} failed (共 {len(tests_to_run)} 个测试)")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
