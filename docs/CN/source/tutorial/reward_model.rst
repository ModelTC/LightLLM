奖励模型部署配置
============================

LightLLM 支持多种奖励模型的推理，用于评估对话质量和生成奖励分数。目前支持的奖励模型包括 InternLM2 Reward 和 Qwen2 Reward 等。

基本启动命令
------------

.. code-block:: bash

    python -m lightllm.server.api_server \
    --port 8080 \
    --model_dir ${MODEL_PATH} \
    --trust_remote_code \
    --use_reward_model # 启用奖励模型功能（必需参数）

测试示例
--------

Python 测试代码
^^^^^^^^^^^^^^^

.. code-block:: python

    import json
    import requests

    # InternLM2 Reward 测试
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
        print(f"奖励分数: {result['score']}")
        print(f"输入token数: {result['prompt_tokens']}")
    else:
        print(f"错误: {response.status_code}, {response.text}")

cURL 测试命令
^^^^^^^^^^^^^

.. code-block:: bash

    curl http://localhost:8000/get_score \
         -H "Content-Type: application/json" \
         -d '{
           "chat": "<|im_start|>user\nHello! What is AI?<|im_end|>\n<|im_start|>assistant\nAI stands for Artificial Intelligence, which refers to the simulation of human intelligence in machines.<|im_end|>\n<|reward|>",
           "parameters": {
             "frequency_penalty": 1
           }
         }'