Examples
================

Qwen2-0.5B
^^^^^^^^^^^^^^^^^^^^^

**Launching Server**

.. code-block:: console

    $ python -m lightllm.server.api_server --model_dir ~/models/Qwen2-0.5B  \
    $                                       --host 0.0.0.0                  \
    $                                       --port 8080                     \
    $                                       --tp 1                          \
    $                                       --max_total_token_num 120000    \
    $                                       --trust_remote_code             \
    $                                       --eos_id 151643

**Test Server**


.. code-block:: console

    $ curl http://localhost:8080/generate \
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

**Launching Server**

.. code-block:: console

    $ python -m lightllm.server.api_server --model_dir ~/models/Qwen-VL-Chat  \
    $                                       --host 0.0.0.0                  \
    $                                       --port 8080                     \
    $                                       --tp 1                          \
    $                                       --max_total_token_num 120000    \
    $                                       --trust_remote_code             \
    $                                       --enable_multimodal

**Test Server**

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

        url = "http://127.0.0.1:8080/generate"
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, data=json.dumps(data))
        return response

    query = """
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    <img></img>
    what is this?<|im_end|>
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

**Launching Server**

.. code-block:: console

    $ python -m lightllm.server.api_server --model_dir ~/models/llama2-70b-chat  \
    $                                       --host 0.0.0.0                       \
    $                                       --port 8080                          \
    $                                       --tp 4                               \
    $                                       --max_total_token_num 120000         

.. tip::

    :code:`--tp` is 4, which means using four cards for tensor parallelism.

**Test Server**

.. code-block:: console

    $ curl http://localhost:8080/generate \
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

**Launching Server**

.. code-block:: console

    $ python -m lightllm.server.api_server --model_dir ~/models/internlm2-1_8b  \
    $                                       --host 0.0.0.0                       \
    $                                       --port 8080                          \
    $                                       --tp 1                               \
    $                                       --max_total_token_num 120000         \
    $                                       --splitfuse_mode                     \
    $                                       --trust_remote_code               

.. tip::

    ``--splitfuse_mode`` Indicates the use of splitfuse for acceleration.


**Test Server**

.. code-block:: console

    $ curl http://localhost:8080/generate \
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

**Launching Server**

.. code-block:: console

    $ python -m lightllm.server.api_server --model_dir ~/models/internlm2-1_8b-reward  \
    $                                       --host 0.0.0.0                       \
    $                                       --port 8080                          \
    $                                       --tp 1                               \
    $                                       --max_total_token_num 120000         \
    $                                       --use_reward_model                    \
    $                                       --trust_remote_code               

.. tip::

    ``--use_reward_model`` Indicates options that must be turned on to use the reward model.


**Test Server**

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