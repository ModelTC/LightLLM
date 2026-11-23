.. _reasoning_parser:

Reasoning Parser
================

LightLLM supports parsing reasoning model's thinking process, separating internal reasoning from final answers to improve AI system transparency.

Supported Models
----------------

DeepSeek-R1
~~~~~~~~~~~

**Parser**: ``deepseek-r1``

**Format**:

.. code-block:: text

    <think>
    Reasoning process...
    </think>
    Final answer

**Features**: Forced reasoning mode, some variants may omit ``<think>`` opening tag

**Startup**:

.. code-block:: bash

    python -m lightllm.server.api_server \
        --model_dir /path/to/DeepSeek-R1 \
        --reasoning_parser deepseek-r1 \
        --tp 8

DeepSeek-V3
~~~~~~~~~~~

**Parser**: ``deepseek-v3``

**Format**: Same as Qwen3

**Startup**:

.. code-block:: bash

    python -m lightllm.server.api_server \
        --model_dir /path/to/DeepSeek-V3 \
        --reasoning_parser deepseek-v3 \
        --tp 8

**Request Config**:

.. code-block:: python

    data = {
        "chat_template_kwargs": {"thinking": True}  # Enable reasoning
    }

Qwen3
~~~~~

**Parser**: ``qwen3``

**Format**: ``<think>Reasoning content</think>Answer``

**Features**: Optional reasoning mode, supports dynamic switching

.. code-block:: python

    # Enable reasoning
    data = {"chat_template_kwargs": {"enable_thinking": True}}

GPT-OSS
~~~~~~~

**Parser**: ``gpt-oss``

**Format**:

.. code-block:: xml

    <|start|><|channel|>analysis<|message|>
    Reasoning analysis...
    <|end|>
    <|channel|>final<|message|>
    Final answer
    <|return|>

**Features**: Complex state machine parsing, supports multiple channels (analysis, commentary, final)

Basic Usage
-----------

Non-Streaming
~~~~~~~~~~~~~

.. code-block:: python

    import requests
    import json

    url = "http://localhost:8088/v1/chat/completions"
    data = {
        "model": "deepseek-r1",
        "messages": [
            {"role": "user", "content": "How many 'r's in 'strawberry'?"}
        ],
        "max_tokens": 2000,
        "separate_reasoning": True,  # Separate reasoning content
        "chat_template_kwargs": {"enable_thinking": True}
    }

    response = requests.post(url, json=data).json()
    message = response["choices"][0]["message"]

    print("Reasoning:", message.get("reasoning_content"))
    print("Answer:", message.get("content"))

Streaming
~~~~~~~~~

.. code-block:: python

    data = {
        "model": "deepseek-r1",
        "messages": [{"role": "user", "content": "Explain quantum entanglement"}],
        "stream": True,
        "separate_reasoning": True,
        "stream_reasoning": True  # Stream reasoning content in real-time
    }

    response = requests.post(url, json=data, stream=True)

    for line in response.iter_lines():
        if line and line.startswith(b"data: "):
            data_str = line[6:].decode('utf-8')
            if data_str == '[DONE]':
                break

            chunk = json.loads(data_str)
            delta = chunk["choices"][0]["delta"]

            # Reasoning content
            if "reasoning_content" in delta:
                print(delta["reasoning_content"], end="", flush=True)

            # Answer content
            if "content" in delta:
                print(delta["content"], end="", flush=True)

Response Format
---------------

**Non-Streaming**:

.. code-block:: json

    {
        "choices": [{
            "message": {
                "content": "Final answer",
                "reasoning_content": "Reasoning process"
            }
        }]
    }

**Streaming**:

.. code-block:: json

    // Reasoning chunk
    {"choices": [{"delta": {"reasoning_content": "Reasoning fragment"}}]}

    // Answer chunk
    {"choices": [{"delta": {"content": "Answer fragment"}}]}

When ``--reasoning_parser`` is enabled, Chat Completions usage reports the
number of generated reasoning tokens (excluding reasoning delimiters):

.. code-block:: json

    {
        "usage": {
            "completion_tokens": 128,
            "completion_tokens_details": {"reasoning_tokens": 96}
        }
    }

The Responses API exposes the same count as
``usage.output_tokens_details.reasoning_tokens``.

Advanced Features
-----------------

Dynamic Reasoning Mode
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Enable reasoning
    data = {
        "chat_template_kwargs": {"enable_thinking": True},
        "separate_reasoning": True
    }

    # Disable reasoning
    data = {
        "chat_template_kwargs": {"enable_thinking": False}
    }

Control Reasoning Display
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Hide reasoning streaming
    data = {
        "separate_reasoning": True,
        "stream_reasoning": False  # reasoning_content field still exists
    }

    # Merge reasoning and answer
    data = {
        "separate_reasoning": False  # Merged in content field
    }

Integration with Tool Calling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

    # Get reasoning, tool calls, and answer simultaneously
    print("Reasoning:", message.get("reasoning_content"))
    print("Tools:", message.get("tool_calls"))
    print("Answer:", message.get("content"))

Multi-Turn Reasoning
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    messages = [{"role": "user", "content": "What is a prime number?"}]

    # First turn
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

    # Second turn
    messages.append({"role": "user", "content": "Is 17 a prime number?"})
    response2 = requests.post(url, json={
        "messages": messages,
        "separate_reasoning": True
    }).json()

Configuration
-------------

**separate_reasoning** (bool, default: True)
  Whether to separate reasoning content into ``reasoning_content`` field

**stream_reasoning** (bool, default: False)
  Whether to stream reasoning content in real-time

**reasoning_effort** (string, optional)
  Accepted reasoning level: ``none``, ``minimal``, ``low``, ``medium``, ``high``, ``xhigh``, or ``max``.
  Support for non-``none`` levels depends on the model's chat template.
  ``none`` automatically disables thinking for compatible chat templates. Explicit
  ``thinking`` or ``enable_thinking`` values in ``chat_template_kwargs`` take priority.

**chat_template_kwargs** (object)
  - ``enable_thinking``: Enable reasoning (Qwen3, GLM45)
  - ``thinking``: Enable reasoning (DeepSeek-V3)

**thinking_token_budget** (integer, optional)
  Enabled by default. Finite budgets require a reasoning parser (which may be
  auto-detected). Use ``--disable_thinking_budget`` to turn the feature off.
  The parser and the model's tokenizer determine the reasoning-end sequence.
  The ``gpt-oss`` parser does not support finite budgets: any value >= 0 returns
  HTTP 400 on Chat and native generation interfaces, including P/D forwarding.
  Omit the budget or use ``-1`` with GPT-OSS. Its final-header suffix cannot be
  used to force a complete Harmony message transition.
  Finite budgets also return HTTP 400 when the switch is off.
  For P/D deployments, keep it enabled on the Master, Prefill, and Decode services.
  Maximum generated thinking tokens before forcing the parser's answer-transition
  token sequence (``</think>`` for Qwen3/Qwen3.5). Omit it or use ``-1`` for
  unlimited thinking; ``0`` transitions immediately to the answer. Explicitly
  disabling thinking makes the budget inactive. A reasoning parser is required.
  For example, pass ``extra_body={"thinking_token_budget": 1024}`` with the OpenAI SDK.
  The budget works with streaming, MTP, and JSON Schema/grammar constraints;
  answer constraints activate after the transition. ``max_completion_tokens``
  still limits the entire output, including the transition and answer.
  Finite budgets cannot be combined with ``--target_vocab_topk_sampling``.
  On ``/generate``, put the budget and ``guided_reasoning_end`` token ID sequence
  in ``parameters`` and configure a reasoning parser.
  For custom markers during PD continuation, see :doc:`known_constraints`.

**--reasoning_parser** (startup parameter)
  Specify parser type: ``deepseek-r1``, ``qwen3``, ``glm45``, ``gpt-oss``, etc.

Common Issues
-------------

**Reasoning content not separated**
  Check ``--reasoning_parser``, ``separate_reasoning: true``, ``chat_template_kwargs``

**Model not generating reasoning**
  Confirm model supports reasoning mode, check if reasoning parameters are enabled

**Incomplete streaming**
  Process all chunks, wait for ``[DONE]`` signal

**Conflict with tool calling**
  Use latest version (includes PR #1158), configure parameters correctly

Performance
-----------

**Token Consumption**: Reasoning mode may increase token usage by 3-5x

**Latency Impact**: TTFB may increase from 200ms to 800ms

**Optimization**:
- Use ``stream_reasoning: true`` to reduce perceived latency
- Disable reasoning mode for non-critical tasks

Technical Details
-----------------

**Core Files**:
- ``lightllm/server/reasoning_parser.py`` - Parser implementation
- ``lightllm/server/api_openai.py`` - API integration
- ``test/test_api/test_openai_api.py`` - Test examples

**Related PRs**:
- PR #1154: Add reasoning parser
- PR #1158: Function call in reasoning content support

References
----------

- DeepSeek-R1 Technical Report
- LightLLM GitHub: https://github.com/ModelTC/lightllm
