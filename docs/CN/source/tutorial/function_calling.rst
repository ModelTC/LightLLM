.. _function_calling:

工具调用（Function Calling）
============================

LightLLM 支持多种主流模型的工具调用功能，提供 OpenAI 兼容的 API。

支持的模型
----------

Qwen2.5/Qwen3
~~~~~~~~~~~~~

**解析器**: ``qwen25``

**格式**:

.. code-block:: xml

    <tool_call>
    {"name": "function_name", "arguments": {"param": "value"}}
    </tool_call>

**启动**:

.. code-block:: bash

    python -m lightllm.server.api_server \
        --model_dir /path/to/qwen2.5 \
        --tool_call_parser qwen25 \
        --tp 1

Llama 3.2
~~~~~~~~~

**解析器**: ``llama3``

**格式**: ``<|python_tag|>{"name": "func", "arguments": {...}}``

**启动**:

.. code-block:: bash

    python -m lightllm.server.api_server \
        --model_dir /path/to/llama-3.2 \
        --tool_call_parser llama3 \
        --tp 1

Mistral
~~~~~~~

**解析器**: ``mistral``

**格式**: ``[TOOL_CALLS] [{"name": "func", "arguments": {...}}, ...]``

DeepSeek-V3
~~~~~~~~~~~

**解析器**: ``deepseekv3``

**格式**:

.. code-block:: xml

    <｜tool▁calls▁begin｜>
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>func_name
    ```json
    {"param": "value"}
    ```
    <｜tool▁call▁end｜>
    <｜tool▁calls▁end｜>

DeepSeek-V3.1
~~~~~~~~~~~~~

**解析器**: ``deepseekv31``

**格式**: 简化的 V3 格式，参数直接内联，无代码块包围

基本使用
--------

定义工具
~~~~~~~~

.. code-block:: python

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]

非流式调用
~~~~~~~~~~

.. code-block:: python

    import requests
    import json

    url = "http://localhost:8088/v1/chat/completions"
    data = {
        "model": "model_name",
        "messages": [
            {"role": "user", "content": "北京今天天气怎么样？"}
        ],
        "tools": tools,
        "tool_choice": "auto"  # "auto" | "none" | "required"
    }

    response = requests.post(url, json=data).json()
    message = response["choices"][0]["message"]

    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            print(f"工具: {tc['function']['name']}")
            print(f"参数: {tc['function']['arguments']}")

流式调用
~~~~~~~~

.. code-block:: python

    data = {
        "model": "model_name",
        "messages": [{"role": "user", "content": "查询北京和上海的天气"}],
        "tools": tools,
        "stream": True
    }

    response = requests.post(url, json=data, stream=True)
    tool_calls = {}

    for line in response.iter_lines():
        if line and line.startswith(b"data: "):
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0]["delta"]

            if delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    idx = tc.get("index", 0)
                    if idx not in tool_calls:
                        tool_calls[idx] = {"function": {"name": "", "arguments": ""}}

                    if tc["function"].get("name"):
                        tool_calls[idx]["function"]["name"] = tc["function"]["name"]
                    if tc["function"].get("arguments"):
                        tool_calls[idx]["function"]["arguments"] += tc["function"]["arguments"]

多轮对话
~~~~~~~~

.. code-block:: python

    # 1. 用户提问
    messages = [{"role": "user", "content": "北京天气如何？"}]

    # 2. 模型调用工具
    response1 = requests.post(url, json={
        "messages": messages,
        "tools": tools
    }).json()

    tool_call = response1["choices"][0]["message"]["tool_calls"][0]
    messages.append(response1["choices"][0]["message"])

    # 3. 返回工具结果
    weather_result = {"temperature": 15, "condition": "晴朗"}
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "name": tool_call["function"]["name"],
        "content": json.dumps(weather_result, ensure_ascii=False)
    })

    # 4. 生成最终回答
    response2 = requests.post(url, json={"messages": messages}).json()
    print(response2["choices"][0]["message"]["content"])

高级功能
--------

并行工具调用
~~~~~~~~~~~~

.. code-block:: python

    data = {
        "messages": messages,
        "tools": tools,
        "parallel_tool_calls": True  # 启用并行调用
    }

强制调用特定工具
~~~~~~~~~~~~~~~~

.. code-block:: python

    data = {
        "tools": tools,
        "tool_choice": {
            "type": "function",
            "function": {"name": "get_weather"}
        }
    }

与推理模型集成
~~~~~~~~~~~~~~

.. code-block:: python

    data = {
        "model": "deepseek-r1",
        "tools": tools,
        "chat_template_kwargs": {"enable_thinking": True},
        "separate_reasoning": True  # 分离推理内容
    }

    response = requests.post(url, json=data).json()
    message = response["choices"][0]["message"]

    print("推理:", message.get("reasoning_content"))
    print("工具调用:", message.get("tool_calls"))

常见问题
--------

**工具调用未触发**
  检查 ``--tool_call_parser`` 参数和工具描述是否清晰

**参数解析错误**
  确认使用了正确的解析器，检查模型输出格式

**流式模式不完整**
  正确处理所有 chunks，使用 ``index`` 字段组装多个工具调用

**与推理模型集成失败**
  确保使用最新版本，正确配置 ``separate_reasoning`` 和 ``chat_template_kwargs``

技术细节
--------

**核心文件**:
- ``lightllm/server/function_call_parser.py`` - 解析器实现
- ``lightllm/server/api_openai.py`` - API 集成
- ``lightllm/server/build_prompt.py`` - 工具注入
- ``test/test_api/test_openai_api.py`` - 测试示例

**相关 PR**:
- PR #1158: 支持推理内容中的函数调用

参考资料
--------

- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
- JSON Schema: https://json-schema.org/
- LightLLM GitHub: https://github.com/ModelTC/lightllm
