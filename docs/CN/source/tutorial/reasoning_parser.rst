.. _reasoning_parser:

思考解析（Reasoning Parser）
=============================

LightLLM 支持推理模型的思考过程解析，将模型内部推理与最终答案分离，提高 AI 系统透明度。

支持的模型
----------

DeepSeek-R1
~~~~~~~~~~~

**解析器**: ``deepseek-r1``

**格式**:

.. code-block:: text

    <think>
    推理过程...
    </think>
    最终答案

**特点**: 强制推理模式，部分变体可能省略 ``<think>`` 起始标签

**启动**:

.. code-block:: bash

    python -m lightllm.server.api_server \
        --model_dir /path/to/DeepSeek-R1 \
        --reasoning_parser deepseek-r1 \
        --tp 8 \
        --llm_prefill_att_backend fa3 \
        --llm_decode_att_backend fa3

DeepSeek-V3
~~~~~~~~~~~

**解析器**: ``deepseek-v3``

**格式**: 与 Qwen3 相同

**启动**:

.. code-block:: bash

    python -m lightllm.server.api_server \
        --model_dir /path/to/DeepSeek-V3 \
        --reasoning_parser deepseek-v3 \
        --tp 8

**请求配置**:

.. code-block:: python

    data = {
        "chat_template_kwargs": {"thinking": True}  # 启用推理
    }

Qwen3
~~~~~

**解析器**: ``qwen3``

**格式**: ``<think>推理内容</think>回答``

**特点**: 可选推理模式，支持动态切换

.. code-block:: python

    # 启用推理
    data = {"chat_template_kwargs": {"enable_thinking": True}}

GPT-OSS
~~~~~~~

**解析器**: ``gpt-oss``

**格式**:

.. code-block:: xml

    <|start|><|channel|>analysis<|message|>
    推理分析...
    <|end|>
    <|channel|>final<|message|>
    最终回答
    <|return|>

**特点**: 复杂状态机解析，支持多通道（analysis, commentary, final）

基本使用
--------

非流式
~~~~~~

.. code-block:: python

    import requests
    import json

    url = "http://localhost:8088/v1/chat/completions"
    data = {
        "model": "deepseek-r1",
        "messages": [
            {"role": "user", "content": "单词 'strawberry' 中有多少个字母 'r'?"}
        ],
        "max_tokens": 2000,
        "separate_reasoning": True,  # 分离推理内容
        "chat_template_kwargs": {"enable_thinking": True}
    }

    response = requests.post(url, json=data).json()
    message = response["choices"][0]["message"]

    print("推理:", message.get("reasoning_content"))
    print("答案:", message.get("content"))

流式
~~~~

.. code-block:: python

    data = {
        "model": "deepseek-r1",
        "messages": [{"role": "user", "content": "解释量子纠缠"}],
        "stream": True,
        "separate_reasoning": True,
        "stream_reasoning": True  # 实时流式传输推理内容
    }

    response = requests.post(url, json=data, stream=True)

    for line in response.iter_lines():
        if line and line.startswith(b"data: "):
            data_str = line[6:].decode('utf-8')
            if data_str == '[DONE]':
                break

            chunk = json.loads(data_str)
            delta = chunk["choices"][0]["delta"]

            # 推理内容
            if "reasoning_content" in delta:
                print(delta["reasoning_content"], end="", flush=True)

            # 答案内容
            if "content" in delta:
                print(delta["content"], end="", flush=True)

响应格式
--------

**非流式**:

.. code-block:: json

    {
        "choices": [{
            "message": {
                "content": "最终答案",
                "reasoning_content": "推理过程"
            }
        }]
    }

**流式**:

.. code-block:: json

    // 推理块
    {"choices": [{"delta": {"reasoning_content": "推理片段"}}]}

    // 答案块
    {"choices": [{"delta": {"content": "答案片段"}}]}

高级功能
--------

动态切换推理模式
~~~~~~~~~~~~~~~~

.. code-block:: python

    # 启用推理
    data = {
        "chat_template_kwargs": {"enable_thinking": True},
        "separate_reasoning": True
    }

    # 禁用推理
    data = {
        "chat_template_kwargs": {"enable_thinking": False}
    }

控制推理显示
~~~~~~~~~~~~

.. code-block:: python

    # 隐藏推理流式传输
    data = {
        "separate_reasoning": True,
        "stream_reasoning": False  # reasoning_content 字段仍存在
    }

    # 合并推理和答案
    data = {
        "separate_reasoning": False  # 推理和答案合并在 content 中
    }

与工具调用集成
~~~~~~~~~~~~~~

.. code-block:: python

    data = {
        "model": "deepseek-r1",
        "tools": tools,
        "tool_choice": "auto",
        "separate_reasoning": True,
        "chat_template_kwargs": {"enable_thinking": True}
    }

    response = requests.post(url, json=data).json()
    message = response["choices"][0]["message"]

    # 同时获得推理、工具调用和答案
    print("推理:", message.get("reasoning_content"))
    print("工具:", message.get("tool_calls"))
    print("答案:", message.get("content"))

多轮推理对话
~~~~~~~~~~~~

.. code-block:: python

    messages = [{"role": "user", "content": "什么是质数？"}]

    # 第一轮
    response1 = requests.post(url, json={
        "messages": messages,
        "separate_reasoning": True
    }).json()

    message1 = response1["choices"][0]["message"]
    messages.append({
        "role": "assistant",
        "content": message1["content"],
        "reasoning_content": message1.get("reasoning_content")
    })

    # 第二轮
    messages.append({"role": "user", "content": "17 是质数吗？"})
    response2 = requests.post(url, json={
        "messages": messages,
        "separate_reasoning": True
    }).json()

配置参数
--------

**separate_reasoning** (布尔, 默认 True)
  是否分离推理内容到 ``reasoning_content`` 字段

**stream_reasoning** (布尔, 默认 False)
  是否实时流式传输推理内容

**chat_template_kwargs** (对象)
  - ``enable_thinking``: 启用推理（Qwen3, GLM45）
  - ``thinking``: 启用推理（DeepSeek-V3）

**--reasoning_parser** (启动参数)
  指定解析器类型：``deepseek-r1``, ``qwen3``, ``glm45``, ``gpt-oss`` 等

常见问题
--------

**推理内容未分离**
  检查 ``--reasoning_parser``, ``separate_reasoning: true``, ``chat_template_kwargs``

**模型不生成推理**
  确认模型支持推理模式，检查是否启用了推理参数

**流式模式不完整**
  处理所有 chunks，等待 ``[DONE]`` 信号

**与工具调用冲突**
  使用最新版本（包含 PR #1158），正确配置参数

性能考虑
--------

**Token 消耗**: 推理模式可能增加 3-5 倍 token 消耗

**延迟影响**: TTFB 可能从 200ms 增加到 800ms

**优化建议**:
- 使用 ``stream_reasoning: true`` 降低感知延迟
- 非关键任务禁用推理模式

技术细节
--------

**核心文件**:
- ``lightllm/server/reasoning_parser.py`` - 解析器实现
- ``lightllm/server/api_openai.py`` - API 集成
- ``test/test_api/test_openai_api.py`` - 测试示例

**相关 PR**:
- PR #1154: 添加推理解析器
- PR #1158: 推理内容中的函数调用支持

参考资料
--------

- DeepSeek-R1 技术报告
- LightLLM GitHub: https://github.com/ModelTC/lightllm
