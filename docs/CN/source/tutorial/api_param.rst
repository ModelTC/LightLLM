接口调用详解
==========================


:code:`GET /health`
~~~~~~~~~~~~~~~~~~~~
:code:`HEAD /health`
~~~~~~~~~~~~~~~~~~~~
:code:`GET /healthz`
~~~~~~~~~~~~~~~~~~~~

获取当前的服务器的运行状态

单机多进程 PD Master 的公开入口汇总 HAProxy 对全部 worker 的周期健康探测，任一
worker 探测失败时返回 HTTP 503；直接访问 worker 端口时返回该进程的本地状态。

**调用示例**： 

.. code-block:: console

    $ curl http://0.0.0.0:8080/health


**输出示例**：

.. code-block:: python

    {"message":"Ok"}



:code:`GET /token_load`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

获取当前的服务器使用token的情况

**调用示例**： 

.. code-block:: console

    $ curl http://0.0.0.0:8080/token_load


**输出示例**：

.. code-block:: python

    {"current_load":0.0,"logical_max_load":0.0,"dynamic_max_load":0.0}


:code:`POST /generate`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

调用模型实现文本补全

**调用示例**： 

.. code-block:: console

    $ curl http://localhost:8080/generate \
    $ -H "Content-Type: application/json" \
    $ -d '{
    $      "inputs": "What is AI?",
    $      "parameters":{
    $        "max_new_tokens":17,
    $        "frequency_penalty":1
    $      },
    $      "multimodal_params":{}
    $     }'


**输出示例**：

.. code-block:: python

    {"generated_text": [" What is the difference between AI and ML? What are the differences between AI and ML"], "count_output_tokens": 17, "finish_reason": "length", "prompt_tokens": 4}


:code:`POST /generate_stream`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

流式返回文本补全结果


**调用示例**： 

.. code-block:: console

    $ curl http://localhost:8080/generate_stream \
    $ -H "Content-Type: application/json" \
    $ -d '{
    $      "inputs": "What is AI?",
    $      "parameters":{
    $        "max_new_tokens":17,
    $        "frequency_penalty":1
    $      },
    $      "multimodal_params":{}
    $     }'

**输出示例**：

::

    data:{"token": {"id": 3555, "text": " What", "logprob": -1.8383026123046875, "special": false, "count_output_tokens": 1, "prompt_tokens": 4}, "generated_text": null, "finished": false, "finish_reason": null, "details": null}

    data:{"token": {"id": 374, "text": " is", "logprob": -0.59185391664505, "special": false, "count_output_tokens": 2, "prompt_tokens": 4}, "generated_text": null, "finished": false, "finish_reason": null, "details": null}

    data:{"token": {"id": 279, "text": " the", "logprob": -1.5594439506530762, "special": false, "count_output_tokens": 3, "prompt_tokens": 4}, "generated_text": null, "finished": true, "finish_reason": "length", "details": null}


停止条件
~~~~~~~~~~~~

``/generate`` 和 ``/generate_stream`` 使用 ``parameters.stop_sequences`` 设置
停止字符串，可以传入字符串或字符串列表。``parameters.include_stop_str_in_output``
默认为 ``false``，返回文本会移除匹配到的 stop 字符串及同一 token 中其后的文本；
设为 ``true`` 则保留 stop 字符串。此参数只影响字符串 stop。

字符串保留原文，不再编码成 token ID 序列。字符串 stop 统一由 HTTP 输出层匹配和截断；
``ALL_TEXT`` 匹配全部生成文本，``CONTENT_ONLY`` 跳过 reasoning、仅匹配回答。

也可以显式传入 token ID 序列，例如 ``"stop_sequences": [[100, 101]]``，其中
ID 应替换为模型实际使用的值。此类 stop 在推理层和 HTTP 输出层独立匹配，包括 reasoning 阶段，
不受 ``stop_str_match_mode`` 或 ``LIGHTLLM_STOP_IN_REASONING`` 影响。
HTTP 输出层先匹配 token ID stop，再匹配字符串 stop；各子请求独立保留真实输出 ID 的尾部，
保证跨 P/D 容量分段的序列也能命中，包括解码文本为空的特殊 token。
命中时保留完整 token 文本和 metadata，只将结束状态改为 stop；用户 prompt 的 IDs 不参与匹配。
``include_stop_str_in_output`` 不改变 token ID stop 的输出行为。
字符串和 token ID 序列可以混合配置，例如 ``["END", [100, 101]]``。
RL、返回 routed experts 和 prompt logprobs 仅支持普通模式。此类请求命中 stop 时仍通知后端停止，
字符串命中包按 ``include_stop_str_in_output`` 截断文本，但保留后端结束状态；后续包原样放行，
直到后端结束包返回最终 metadata。因此命中后仍可能返回在途 token，最终结束原因由后端决定。
PD 分离模式启动时拒绝这些功能开关；请求参数不能启用对应的服务端功能。

.. list-table:: stop 匹配范围
   :header-rows: 1

   * - ``stop_str_match_mode``
     - 字符串 stop
     - 显式 token ID stop
   * - ``1`` / ``ALL_TEXT``（默认）
     - 全部生成文本
     - 全部生成 token
   * - ``2`` / ``CONTENT_ONLY``
     - 仅回答，跳过 reasoning
     - 全部生成 token

流式和非流式遵循相同设置。输出尾部匹配任意 stop 字符串的前缀时，流式输出会暂存
覆盖该前缀的完整 token，直到能够确认是否命中 stop，防止跨 token 的 stop 片段提前发出。
请求结束时会返回未匹配的尾部。token ID、logprob、输出 token 计数和结束原因的含义保持不变。

``/generate`` 和 ``/generate_stream`` 默认对全部生成文本匹配字符串 stop，包含 reasoning 文本，
不因服务配置了 reasoning parser 而自动区分阶段。OpenAI ``/v1/chat/completions``
的 reasoning / content 停止范围设置见 :ref:`OpenAI 接口示例 <openai_api>`。


:code:`POST /get_score`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
reward 类模型，获取对话分数

**调用示例**： 

.. code-block:: python

    import json
    import requests

    query = "<|im_start|>user\nHello! What's your name?<|im_end|>\n<|im_start|>assistant\nMy name is InternLM2! A helpful AI assistant. What can I do for you?<|im_end|>\n<|reward|>"

    url = "http://127.0.0.1:8080/get_score"
    headers = {'Content-Type': 'application/json'}

    data = {
        "chat": query,
        "parameters": {
            "frequency_penalty":1
        }
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        print(f"Result: {response.json()}")
    else:
        print(f"Error: {response.status_code}, {response.text}")

**输出示例**：

::

    Result: {'score': 0.4892578125, 'prompt_tokens': 39, 'finish_reason': 'stop'}