Reward Model Deployment Configuration
=====================================

LightLLM supports inference for various reward models, used for evaluating conversation quality and generating reward scores. Currently supported reward models include InternLM2 Reward and Qwen2 Reward, etc.

Basic Launch Command
---------------------

.. code-block:: bash

    python -m lightllm.server.api_server \
    --port 8080 \
    --model_dir ${MODEL_PATH} \
    --trust_remote_code \
    --use_reward_model # Enable reward model functionality (required parameter)

Testing Examples
----------------

Python Testing Code
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import json
    import requests

    # InternLM2 Reward test
    query = "<|im_start|>user\nHello! What's your name?<|im_end|>\n<|im_start|>assistant\nMy name is InternLM2! A helpful AI assistant. What can I do for you?<|im_end|>\n<|reward|>"

    url = "http://127.0.0.1:8000/get_score"
    headers = {'Content-Type': 'application/json'}

    data = {
        "chat": query,
        "parameters": {
            "frequency_penalty": 1
        }
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        result = response.json()
        print(f"Reward score: {result['score']}")
        print(f"Input tokens: {result['prompt_tokens']}")
    else:
        print(f"Error: {response.status_code}, {response.text}")

cURL Testing Command
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    curl http://localhost:8000/get_score \
         -H "Content-Type: application/json" \
         -d '{
           "chat": "<|im_start|>user\nHello! What is AI?<|im_end|>\n<|im_start|>assistant\nAI stands for Artificial Intelligence, which refers to the simulation of human intelligence in machines.<|im_end|>\n<|reward|>",
           "parameters": {
             "frequency_penalty": 1
           }
         }' 