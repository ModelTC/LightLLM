接口调用详解
==========================


:code:`GET /health`
~~~~~~~~~~~~~~~~~~~~
:code:`HEAD /health`
~~~~~~~~~~~~~~~~~~~~
:code:`GET /healthz`
~~~~~~~~~~~~~~~~~~~~

获取当前的服务器的运行状态

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


停止字符串
~~~~~~~~~~~~

``/generate`` 和 ``/generate_stream`` 使用 ``parameters.stop_sequences`` 设置
停止字符串，可以传入字符串或字符串列表。``parameters.include_stop_str_in_output``
默认为 ``false``，返回文本会移除匹配到的 stop 字符串及同一 token 中其后的文本；
设为 ``true`` 则保留 stop 字符串。此参数只影响字符串 stop，token ID 序列的行为保持不变。

流式和非流式遵循相同设置。输出尾部匹配任意 stop 字符串的前缀时，流式输出会暂存
覆盖该前缀的完整 token，直到能够确认是否命中 stop，防止跨 token 的 stop 片段提前发出。
请求结束时会返回未匹配的尾部。token ID、logprob、输出 token 计数和结束原因的含义保持不变。

配置 reasoning parser 后，字符串 stop 和 token ID stop 默认都跳过 reasoning，
只作用于回答 content。启动服务前设置 ``LIGHTLLM_STOP_IN_REASONING=1``，可使其在两个阶段
都生效。EOS 和输出 token 上限仍在两个阶段生效。部署说明见 :ref:`OpenAI 接口示例 <openai_api>`。


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