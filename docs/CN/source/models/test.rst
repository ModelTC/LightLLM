启动和测试模型示例
====================

Qwen2-0.5B
^^^^^^^^^^^^^^^^^^^^^

**启动服务**

.. code-block:: console

    $ python -m lightllm.server.api_server --model_dir ~/models/Qwen2-0.5B --trust_remote_code             

**测试服务**


.. code-block:: console

    $ curl http://localhost:8000/generate \
    $      -H "Content-Type: application/json" \
    $      -d '{
    $            "inputs": "What is AI?",
    $            "parameters":{
    $              "max_new_tokens":17, 
    $              "frequency_penalty":1
    $            }
    $           }'


Qwen-VL-Chat
^^^^^^^^^^^^^^^^^

**启动服务**

.. code-block:: console

    $ python -m lightllm.server.api_server 
    $                --model_dir ~/models/Qwen-VL-Chat  \
    $                --trust_remote_code             \
    $                --enable_multimodal

**测试服务**

.. code-block:: python

    import json
    import requests
    import base64

    def run(query, uris):
        images = []
        for uri in uris:
            if uri.startswith("http"):
                images.append({"type": "url", "data": uri})
            else:
                with open(uri, 'rb') as fin:
                    b64 = base64.b64encode(fin.read()).decode("utf-8")
                images.append({'type': "base64", "data": b64})

        data = {
            "inputs": query,
            "parameters": {
                "max_new_tokens": 200,
                # The space before <|endoftext|> is important,
                # the server will remove the first bos_token_id,
                # but QWen tokenizer does not has bos_token_id
                "stop_sequences": [" <|endoftext|>", " <|im_start|>", " <|im_end|>"],
            },
            "multimodal_params": {
                "images": images,
            }
        }

        url = "http://127.0.0.1:8000/generate"
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, data=json.dumps(data))
        return response

    query = """
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    <img></img>
    这是什么？<|im_end|>
    <|im_start|>assistant
    """

    response = run(
        uris = [
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        ],
        query = query
    )

    if response.status_code == 200:
        print(f"Result: {response.json()}")
    else:
        print(f"Error: {response.status_code}, {response.text}")



llama2-70b-chat
^^^^^^^^^^^^^^^^^^^^^^^

**启动服务**

.. code-block:: console

    $ python -m lightllm.server.api_server --model_dir ~/models/llama2-70b-chat --tp 4                               

.. tip::

    :code:`--tp` 为4，表示使用四张卡进行张量并行。

**测试服务**

.. code-block:: console

    $ curl http://localhost:8000/generate \
    $      -H "Content-Type: application/json" \
    $      -d '{
    $            "inputs": "What is LLM?",
    $            "parameters":{
    $              "max_new_tokens":170, 
    $              "frequency_penalty":1
    $            }
    $           }'


internlm2-1_8b
^^^^^^^^^^^^^^^^^^^^^^^

**启动服务**

.. code-block:: console

    $ python -m lightllm.server.api_server 
    $           --model_dir ~/models/internlm2-1_8b  \
    $           --splitfuse_mode                     \
    $           --trust_remote_code               

.. tip::

    ``--splitfuse_mode`` 表示使用splitfuse进行加速。


**测试服务**

.. code-block:: console

    $ curl http://localhost:8000/generate \
    $      -H "Content-Type: application/json" \
    $      -d '{
    $            "inputs": "What is LLM?",
    $            "parameters":{
    $              "max_new_tokens":170, 
    $              "frequency_penalty":1
    $            }
    $           }'


internlm2-1_8b-reward
^^^^^^^^^^^^^^^^^^^^^^^

**启动服务**

.. code-block:: console

    $ python -m lightllm.server.api_server 
    $           --model_dir ~/models/internlm2-1_8b-reward  \
    $           --use_reward_model \
    $           --trust_remote_code               

.. tip::

    ``--use_reward_model`` 表示使用 reward 类模型必须要打开的选项。


**测试服务**

.. code-block:: python

    import json
    import requests

    query = "<|im_start|>user\nHello! What's your name?<|im_end|>\n<|im_start|>assistant\nMy name is InternLM2! A helpful AI assistant. What can I do for you?<|im_end|>\n<|reward|>"

    url = "http://127.0.0.1:8000/get_score"
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