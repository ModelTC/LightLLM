.. _openai_api:

LightLLM OpenAI 接口调用示例
============================

LightLLM 提供了与 OpenAI API 完全兼容的接口，支持所有标准的 OpenAI 功能，包括 function calling。本文档将详细介绍如何使用 LightLLM 的 OpenAI 接口。

基础配置
--------

首先确保 LightLLM 服务已经启动：

.. code-block:: bash

    # 启动 LightLLM 服务
    python -m lightllm.server.api_server \
        --model_dir /path/to/your/model \
        --port 8088 \
        --tp 1

基础对话示例
------------

1. 简单对话
~~~~~~~~~~~

.. code-block:: python

    import requests
    import json

    # 配置
    url = "http://localhost:8088/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    # 请求数据
    data = {
        "model": "your_model_name",
        "messages": [
            {"role": "user", "content": "你好，请介绍一下你自己"}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }

    # 发送请求
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print("回复:", result["choices"][0]["message"]["content"])
    else:
        print("错误:", response.status_code, response.text)

2. 流式对话
~~~~~~~~~~~

.. code-block:: python

    import requests
    import json

    url = "http://localhost:8088/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    data = {
        "model": "your_model_name",
        "messages": [
            {"role": "user", "content": "请写一个关于人工智能的短文"}
        ],
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 1000
    }

    # 流式请求
    response = requests.post(url, headers=headers, json=data, stream=True)
    
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]  # 移除 "data: " 前缀
                    if data_str == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data_str)
                        if chunk['choices'][0]['delta'].get('content'):
                            print(chunk['choices'][0]['delta']['content'], end='', flush=True)
                    except json.JSONDecodeError:
                        continue
    else:
        print("错误:", response.status_code, response.text)

Function Calling 示例
---------------------

LightLLM 支持 OpenAI 的 function calling 功能，提供了三种模型的函数调用解析，启动服务的时候指定 --tool_call_parser 参数来选择。启动服务命令为：

.. code-block:: bash

    python -m lightllm.server.api_server \
        --model_dir /path/to/your/model \
        --port 8088 \
        --tp 1 \
        --tool_call_parser qwen25
    # 可选的参数为 qwen25, llama3, mistral

1. 基础 Function Calling
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import requests
    import json

    url = "http://localhost:8088/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    # 定义函数
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "获取指定城市的当前天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称，例如：北京、上海"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "温度单位"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    # 请求数据
    data = {
        "model": "your_model_name",
        "messages": [
            {"role": "user", "content": "北京今天天气怎么样？"}
        ],
        "tools": tools,
        "tool_choice": "auto",  # 让模型自动决定是否调用函数
        "temperature": 0.7,
        "max_tokens": 1000
    }

    # 发送请求
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        message = result["choices"][0]["message"]
        
        # 检查是否有函数调用
        if message.get("tool_calls"):
            print("模型决定调用函数:")
            for tool_call in message["tool_calls"]:
                print(f"函数名: {tool_call['function']['name']}")
                print(f"参数: {tool_call['function']['arguments']}")
        else:
            print("回复:", message["content"])
    else:
        print("错误:", response.status_code, response.text)

2. 流式 Function Calling
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import requests
    import json

    url = "http://localhost:8088/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "执行数学计算",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "数学表达式"}
                    },
                    "required": ["expression"]
                }
            }
        }
    ]

    data = {
        "model": "your_model_name",
        "messages": [
            {"role": "user", "content": "请计算 25 * 4 + 10 的结果"}
        ],
        "tools": tools,
        "tool_choice": "auto",
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 1000
    }

    response = requests.post(url, headers=headers, json=data, stream=True)
    
    if response.status_code == 200:
        content_buffer = ""
        tool_calls_buffer = []
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk['choices'][0]['delta']
                        
                        # 处理内容
                        if delta.get('content'):
                            content_buffer += delta['content']
                            print(delta['content'], end='', flush=True)
                        
                        # 处理函数调用
                        if delta.get('tool_calls'):
                            for tool_call in delta['tool_calls']:
                                tool_calls_buffer.append(tool_call)
                                print(f"\n[函数调用: {tool_call['function']['name']}]")
                                if tool_call['function'].get('arguments'):
                                    print(f"参数: {tool_call['function']['arguments']}")
                                    
                    except json.JSONDecodeError:
                        continue
    else:
        print("错误:", response.status_code, response.text)
