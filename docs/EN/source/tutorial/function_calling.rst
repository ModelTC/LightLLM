.. _function_calling:

Function Calling
================

LightLLM supports function calling for multiple mainstream models. Provides OpenAI-compatible API.

Supported Models
----------------

Qwen2.5/Qwen3
~~~~~~~~~~~~~

**Parser**: ``qwen25``

**Format**:

.. code-block:: xml

    <tool_call>
    {"name": "function_name", "arguments": {"param": "value"}}
    </tool_call>

**Startup**:

.. code-block:: bash

    python -m lightllm.server.api_server \
        --model_dir /path/to/qwen2.5 \
        --tool_call_parser qwen25 \
        --tp 1

Llama 3.2
~~~~~~~~~

**Parser**: ``llama3``

**Format**: ``<|python_tag|>{"name": "func", "arguments": {...}}``

**Startup**:

.. code-block:: bash

    python -m lightllm.server.api_server \
        --model_dir /path/to/llama-3.2 \
        --tool_call_parser llama3 \
        --tp 1

Mistral
~~~~~~~

**Parser**: ``mistral``

**Format**: ``[TOOL_CALLS] [{"name": "func", "arguments": {...}}, ...]``

DeepSeek-V3
~~~~~~~~~~~

**Parser**: ``deepseekv3``

**Format**:

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

**Parser**: ``deepseekv31``

**Format**: Simplified V3 format, parameters directly inlined without code blocks

Basic Usage
-----------

Define Tools
~~~~~~~~~~~~

.. code-block:: python

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]

Non-Streaming
~~~~~~~~~~~~~

.. code-block:: python

    import requests
    import json

    url = "http://localhost:8088/v1/chat/completions"
    data = {
        "model": "model_name",
        "messages": [
            {"role": "user", "content": "What's the weather in Beijing?"}
        ],
        "tools": tools,
        "tool_choice": "auto"  # "auto" | "none" | "required"
    }

    response = requests.post(url, json=data).json()
    message = response["choices"][0]["message"]

    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            print(f"Tool: {tc['function']['name']}")
            print(f"Args: {tc['function']['arguments']}")

Streaming
~~~~~~~~~

.. code-block:: python

    data = {
        "model": "model_name",
        "messages": [{"role": "user", "content": "Check weather for Beijing and Shanghai"}],
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

Multi-Turn Conversation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # 1. User question
    messages = [{"role": "user", "content": "How's the weather in Beijing?"}]

    # 2. Model calls tool
    response1 = requests.post(url, json={
        "messages": messages,
        "tools": tools
    }).json()

    tool_call = response1["choices"][0]["message"]["tool_calls"][0]
    messages.append(response1["choices"][0]["message"])

    # 3. Return tool result
    weather_result = {"temperature": 15, "condition": "sunny"}
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "name": tool_call["function"]["name"],
        "content": json.dumps(weather_result)
    })

    # 4. Generate final answer
    response2 = requests.post(url, json={"messages": messages}).json()
    print(response2["choices"][0]["message"]["content"])

Advanced Features
-----------------

Parallel Tool Calls
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    data = {
        "messages": messages,
        "tools": tools,
        "parallel_tool_calls": True  # Enable parallel calls
    }

Force Specific Tool
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    data = {
        "tools": tools,
        "tool_choice": {
            "type": "function",
            "function": {"name": "get_weather"}
        }
    }

Integration with Reasoning Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    data = {
        "model": "deepseek-r1",
        "tools": tools,
        "chat_template_kwargs": {"enable_thinking": True},
        "separate_reasoning": True  # Separate reasoning content
    }

    response = requests.post(url, json=data).json()
    message = response["choices"][0]["message"]

    print("Reasoning:", message.get("reasoning_content"))
    print("Tool calls:", message.get("tool_calls"))

Common Issues
-------------

**Tool calls not triggered**
  Check ``--tool_call_parser`` parameter and tool descriptions

**Parameter parsing errors**
  Confirm correct parser is used, check model output format

**Incomplete streaming**
  Process all chunks correctly, use ``index`` field to assemble multiple calls

**Integration with reasoning models fails**
  Use latest version, configure ``separate_reasoning`` and ``chat_template_kwargs``

Technical Details
-----------------

**Core Files**:
- ``lightllm/server/function_call_parser.py`` - Parser implementation
- ``lightllm/server/api_openai.py`` - API integration
- ``lightllm/server/build_prompt.py`` - Tool injection
- ``test/test_api/test_openai_api.py`` - Test examples

**Related PRs**:
- PR #1158: Function call in reasoning content support

References
----------

- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
- JSON Schema: https://json-schema.org/
- LightLLM GitHub: https://github.com/ModelTC/lightllm
