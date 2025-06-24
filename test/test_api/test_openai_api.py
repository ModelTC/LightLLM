#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightLLM OpenAI API test cases

python test_openai_api.py
"""

import requests
import json
import time
from typing import Dict, List, Any, Optional


class LightLLMClient:
    """LightLLM OpenAI API test cases"""

    def __init__(self, base_url: str = "http://localhost:8000", model_name: str = "your_model_name"):
        self.base_url = base_url
        self.model_name = model_name
        self.headers = {"Content-Type": "application/json"}
        self.conversation_history = []

    def simple_chat(self, message: str, **kwargs) -> Dict[str, Any]:
        """简单对话"""
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": message}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
            **kwargs,
        }

        response = requests.post(f"{self.base_url}/v1/chat/completions", headers=self.headers, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")

    def stream_chat(self, message: str, **kwargs):
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": message}],
            "stream": True,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
            **kwargs,
        }

        response = requests.post(f"{self.base_url}/v1/chat/completions", headers=self.headers, json=data, stream=True)

        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            if chunk["choices"][0]["delta"].get("content"):
                                yield chunk["choices"][0]["delta"]["content"]
                        except json.JSONDecodeError:
                            continue
        else:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")

    def function_call(self, message: str, tools: List[Dict], tool_choice: str = "auto", **kwargs) -> Dict[str, Any]:
        """Function calling"""
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": message}],
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
            **kwargs,
        }

        response = requests.post(f"{self.base_url}/v1/chat/completions", headers=self.headers, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")

    def stream_function_call(self, message: str, tools: List[Dict], tool_choice: str = "auto", **kwargs):
        """stream Function calling"""
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": message}],
            "tools": tools,
            "tool_choice": tool_choice,
            "stream": True,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
            **kwargs,
        }

        response = requests.post(f"{self.base_url}/v1/chat/completions", headers=self.headers, json=data, stream=True)

        if response.status_code == 200:
            content_buffer = ""
            tool_calls_buffer = []

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0]["delta"]

                            # 处理内容
                            if delta.get("content"):
                                content_buffer += delta["content"]
                                yield {"type": "content", "data": delta["content"]}

                            # 处理函数调用
                            if delta.get("tool_calls"):
                                for tool_call in delta["tool_calls"]:
                                    tool_calls_buffer.append(tool_call)
                                    yield {"type": "tool_call", "data": tool_call}

                        except json.JSONDecodeError:
                            continue
        else:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")


def test_simple_chat():
    client = LightLLMClient()

    try:
        result = client.simple_chat("你好，请介绍一下你自己")
        print("用户: 你好，请介绍一下你自己")
        print("助手:", result["choices"][0]["message"]["content"])
        print()
    except Exception as e:
        print(f"错误: {e}")
        print("请确保 LightLLM 服务已启动，并检查配置")


def test_stream_chat():
    client = LightLLMClient()

    try:
        print("用户: 请写一个关于人工智能的短文")
        print("助手: ", end="", flush=True)

        for chunk in client.stream_chat("请写一个关于人工智能的短文"):
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"错误: {e}")


def test_function_call():
    client = LightLLMClient()

    # 定义函数
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称，例如：北京、上海"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "温度单位"},
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "执行数学计算",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string", "description": "数学表达式，例如：2+3*4"}},
                    "required": ["expression"],
                },
            },
        },
    ]

    try:
        # 测试天气查询
        print("用户: 北京今天天气怎么样？")
        result = client.function_call("北京今天天气怎么样？", tools)
        message = result["choices"][0]["message"]

        if message.get("tool_calls"):
            print("助手决定调用函数:")
            for tool_call in message["tool_calls"]:
                print(f"  函数名: {tool_call['function']['name']}")
                print(f"  参数: {tool_call['function']['arguments']}")
        else:
            print("助手:", message["content"])
        print()

        # 测试数学计算
        print("用户: 请计算 25 * 4 + 10 的结果")
        result = client.function_call("请计算 25 * 4 + 10 的结果", tools)
        message = result["choices"][0]["message"]

        if message.get("tool_calls"):
            print("助手决定调用函数:")
            for tool_call in message["tool_calls"]:
                print(f"  函数名: {tool_call['function']['name']}")
                print(f"  参数: {tool_call['function']['arguments']}")
        else:
            print("助手:", message["content"])
        print()

    except Exception as e:
        print(f"错误: {e}")


def test_stream_function_call():

    client = LightLLMClient()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    try:
        print("用户: 上海今天天气怎么样？")
        print("助手: ", end="", flush=True)

        for chunk in client.stream_function_call("上海今天天气怎么样？", tools):
            if chunk["type"] == "content":
                print(chunk["data"], end="", flush=True)
            elif chunk["type"] == "tool_call":
                print(f"\n[函数调用: {chunk['data']['function']['name']}]")
                if chunk["data"]["function"].get("arguments"):
                    print(f"参数: {chunk['data']['function']['arguments']}")
        print("\n")

    except Exception as e:
        print(f"错误: {e}")


def main():
    test_simple_chat()
    test_stream_chat()
    test_function_call()
    test_stream_function_call()


if __name__ == "__main__":
    main()
