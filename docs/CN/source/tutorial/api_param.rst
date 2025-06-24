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