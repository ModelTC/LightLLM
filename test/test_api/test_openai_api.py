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

    def completions(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """文本补全"""
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 100),
            **kwargs,
        }

        response = requests.post(f"{self.base_url}/v1/completions", headers=self.headers, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")

    def stream_completions(self, prompt: str, **kwargs):
        """流式文本补全"""
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 100),
            **kwargs,
        }

        response = requests.post(f"{self.base_url}/v1/completions", headers=self.headers, json=data, stream=True)

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
                            if chunk["choices"][0].get("text"):
                                yield chunk["choices"][0]["text"]
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

    def completions_with_tokens(self, token_ids: List[int], **kwargs) -> Dict[str, Any]:
        """使用token数组进行文本补全"""
        data = {
            "model": self.model_name,
            "prompt": token_ids,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 100),
            **kwargs,
        }

        response = requests.post(f"{self.base_url}/v1/completions", headers=self.headers, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")

    def completions_with_multiple_prompts(self, prompts: List[str], **kwargs) -> Dict[str, Any]:
        """使用多个prompt进行文本补全（只处理第一个）"""
        data = {
            "model": self.model_name,
            "prompt": prompts,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 100),
            **kwargs,
        }

        response = requests.post(f"{self.base_url}/v1/completions", headers=self.headers, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")

    def completions_with_logprobs(self, prompt: str, logprobs: int = 5, **kwargs) -> Dict[str, Any]:
        """测试带logprobs的文本补全"""
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "logprobs": logprobs,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 50),
            **kwargs,
        }

        response = requests.post(f"{self.base_url}/v1/completions", headers=self.headers, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")

    def completions_with_echo(self, prompt: str, echo: bool = True, **kwargs) -> Dict[str, Any]:
        """测试带echo参数的文本补全"""
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "echo": echo,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 30),
            **kwargs,
        }

        response = requests.post(f"{self.base_url}/v1/completions", headers=self.headers, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")

    def completions_with_echo_and_logprobs(
        self, prompt: str, echo: bool = True, logprobs: int = 5, **kwargs
    ) -> Dict[str, Any]:
        """测试带echo和logprobs参数的文本补全（重点测试修复后的功能）"""
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "echo": echo,
            "logprobs": logprobs,
            "temperature": kwargs.get("temperature", 0.0),  # 使用0温度以获得一致结果
            "max_tokens": kwargs.get("max_tokens", 20),
            **kwargs,
        }

        response = requests.post(f"{self.base_url}/v1/completions", headers=self.headers, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")

    def completions_logprobs_structure_test(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """专门测试logprobs数据结构的完整性"""
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "logprobs": 3,
            "echo": True,
            "temperature": 0.0,
            "max_tokens": 10,
            **kwargs,
        }

        response = requests.post(f"{self.base_url}/v1/completions", headers=self.headers, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")

    def completions_with_n(self, prompt: str, n: int = 2, **kwargs) -> Dict[str, Any]:
        """测试n参数生成多个候选"""
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "n": n,
            "best_of": n,  # LightLLM要求n == best_of
            "temperature": kwargs.get("temperature", 0.8),
            "max_tokens": kwargs.get("max_tokens", 10),  # 确保max_tokens至少为1
            **kwargs,
        }

        response = requests.post(f"{self.base_url}/v1/completions", headers=self.headers, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")

    def completions_with_stop(self, prompt: str, stop, **kwargs) -> Dict[str, Any]:
        """测试stop参数"""
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stop": stop,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 50),
            **kwargs,
        }

        response = requests.post(f"{self.base_url}/v1/completions", headers=self.headers, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")

    def completions_with_multiple_token_arrays(self, token_arrays: List[List[int]], **kwargs) -> Dict[str, Any]:
        """测试多个token数组的批处理"""
        data = {
            "model": self.model_name,
            "prompt": token_arrays,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 30),
            **kwargs,
        }

        response = requests.post(f"{self.base_url}/v1/completions", headers=self.headers, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")


def test_completions():
    """测试文本补全API"""
    client = LightLLMClient()

    try:
        print("=== 测试文本补全 ===")
        result = client.completions("The capital of France is", max_tokens=50)
        print("提示: The capital of France is")
        print("补全:", result["choices"][0]["text"])
        print(f"用量: {result['usage']}")
        print()
    except Exception as e:
        print(f"错误: {e}")


def test_stream_completions():
    """测试流式文本补全API"""
    client = LightLLMClient()

    try:
        print("=== 测试流式文本补全 ===")
        print("提示: Once upon a time")
        print("补全: ", end="", flush=True)

        for chunk in client.stream_completions("Once upon a time", max_tokens=100):
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"错误: {e}")


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


def test_token_completions():
    """测试使用token数组的文本补全API"""
    client = LightLLMClient()

    try:
        print("=== 测试token数组补全 ===")
        # 示例token数组 (这些是示例值，实际应该用正确的tokenizer)
        token_ids = [2701, 525, 5248, 5754, 4755]  # 示例token
        result = client.completions_with_tokens(token_ids, max_tokens=50)
        print(f"Token IDs: {token_ids}")
        print("补全:", result["choices"][0]["text"])
        print(f"用量: {result['usage']}")
        print()
    except Exception as e:
        print(f"错误: {e}")


def test_multiple_prompts():
    """测试多个prompt的文本补全API（真正的批处理）"""
    client = LightLLMClient()

    try:
        print("=== 测试批处理补全 ===")
        prompts = ["Hello, how are you?", "What is the weather like?", "Tell me a joke"]
        result = client.completions_with_multiple_prompts(prompts, max_tokens=30)
        print(f"发送了 {len(prompts)} 个prompts进行批处理:")

        for i, choice in enumerate(result["choices"]):
            print(f"  {i+1}. 提示: {prompts[choice['index']]}")
            print(f"     补全: {choice['text'].strip()}")
            print(f"     完成原因: {choice['finish_reason']}")

        print(f"总用量: {result['usage']}")
        print()
    except Exception as e:
        print(f"错误: {e}")


def test_logprobs():
    """测试logprobs功能"""
    client = LightLLMClient()

    try:
        print("=== 测试logprobs ===")
        result = client.completions_with_logprobs("The capital of France is", logprobs=5, max_tokens=20)
        print("提示: The capital of France is")
        print("补全:", result["choices"][0]["text"])

        # 检查logprobs结构
        logprobs = result["choices"][0]["logprobs"]
        if logprobs:
            print("Logprobs结构:")
            print(f"  tokens: {logprobs.get('tokens', [])[:5]}...")  # 只显示前5个
            print(f"  token_logprobs: {logprobs.get('token_logprobs', [])[:5]}...")
            print(f"  text_offset: {logprobs.get('text_offset', [])[:5]}...")
            print(f"  top_logprobs: {logprobs.get('top_logprobs', [])[:2]}...")  # 只显示前2个
        print()
    except Exception as e:
        print(f"错误: {e}")


def test_echo():
    """测试echo参数"""
    client = LightLLMClient()

    try:
        print("=== 测试echo参数 ===")

        # 测试echo=True
        result = client.completions_with_echo("Hello world", echo=True, max_tokens=20)
        print("提示: Hello world (echo=True)")
        print("补全:", repr(result["choices"][0]["text"]))
        print()

        # 测试echo=False
        result = client.completions_with_echo("Hello world", echo=False, max_tokens=20)
        print("提示: Hello world (echo=False)")
        print("补全:", repr(result["choices"][0]["text"]))
        print()
    except Exception as e:
        print(f"错误: {e}")


def test_stop_parameter():
    """测试stop参数"""
    client = LightLLMClient()

    try:
        print("=== 测试stop参数 ===")

        # 测试单个stop字符串
        result = client.completions_with_stop("Count: 1, 2, 3, 4", stop="12", max_tokens=50)
        print("提示: Count: 1, 2, 3, 4 (stop='12')")
        print("补全:", repr(result["choices"][0]["text"]))
        print("完成原因:", result["choices"][0]["finish_reason"])
        print()

        # 测试多个stop字符串
        result = client.completions_with_stop("The colors are red, blue, green", stop=["red", "blue"], max_tokens=50)
        print("提示: The colors are red, blue, green (stop=['red', 'blue'])")
        print("补全:", repr(result["choices"][0]["text"]))
        print("完成原因:", result["choices"][0]["finish_reason"])
        print()
    except Exception as e:
        print(f"错误: {e}")


def test_multiple_token_arrays():
    """测试多个token数组的批处理"""
    client = LightLLMClient()

    try:
        print("=== 测试多个token数组批处理 ===")
        token_arrays = [[2701, 525, 5248], [4755, 8394, 1234], [9876, 5432, 1098]]

        result = client.completions_with_multiple_token_arrays(token_arrays, max_tokens=20)
        print(f"发送了 {len(token_arrays)} 个token数组进行批处理:")

        for i, choice in enumerate(result["choices"]):
            print(f"  {i+1}. Token数组: {token_arrays[choice['index']]}")
            print(f"     补全: {choice['text'].strip()}")
            print(f"     完成原因: {choice['finish_reason']}")
        print()
    except Exception as e:
        print(f"错误: {e}")


def test_structured_generation():
    """测试结构化生成功能"""
    client = LightLLMClient()

    try:
        print("=== 测试结构化生成 ===")
        prompt = "请以JSON格式提供一个包含姓名、年龄和职业的人的信息。"

        # 测试JSON生成
        result = client.completions(prompt, max_tokens=150, response_format={"type": "json_object"})
        print("提示:", prompt)
        print("助手:", result["choices"][0]["text"])

        # 测试JSON Schema生成
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "occupation": {"type": "string"},
            },
            "required": ["name", "age", "occupation"],
        }
        result = client.completions(
            prompt,
            max_tokens=150,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "PersonInfo",
                    "description": "包含姓名、年龄和职业的人的信息",
                    "schema": schema,
                },
            },
        )
        print("提示:", prompt)
        print("助手:", result["choices"][0]["text"])

        # 测试/v1/chat/completions端点的JSON生成
        result = client.simple_chat(
            prompt,
            max_tokens=150,
            response_format={"type": "json_object"},
        )
        print("提示:", prompt)
        print("助手:", result["choices"][0]["message"]["content"])

        # 测试/v1/chat/completions端点的JSON Schema生成
        result = client.simple_chat(
            prompt,
            max_tokens=150,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "PersonInfo",
                    "description": "包含姓名、年龄和职业的人的信息",
                    "schema": schema,
                },
            },
        )
        print("提示:", prompt)
        print("助手:", result["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"错误: {e}")


def main():
    # 基础功能测试
    test_completions()
    test_stream_completions()
    test_simple_chat()
    test_stream_chat()
    test_function_call()
    test_stream_function_call()

    # 高级功能测试
    test_token_completions()
    test_multiple_prompts()
    test_multiple_token_arrays()
    test_logprobs()
    test_echo()
    test_stop_parameter()
    test_structured_generation()


if __name__ == "__main__":
    main()
