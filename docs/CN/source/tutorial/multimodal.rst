多模态模型启动配置
============================

LightLLM支持多种多模态模型的推理，下面以InternVL为例，对多模态服务的启动命令进行说明。

基本启动命令
------------

.. code-block:: bash

    INTERNVL_IMAGE_LENGTH=256 \
    LOADWORKER=12 \
    python -m lightllm.server.api_server \
    --port 8080 \
    --tp 2 \
    --model_dir ${MODEL_PATH} \
    --mem_fraction 0.8 \
    --trust_remote_code \
    --enable_multimodal

核心参数说明
------------

环境变量
^^^^^^^^

- **INTERNVL_IMAGE_LENGTH**: 设置InternVL模型的图像token长度，默认为256
- **LOADWORKER**: 设置模型加载的工作进程数

基础服务参数
^^^^^^^^^^^^

- **--port 8080**: API服务器监听端口
- **--tp 2**: 张量并行度(Tensor Parallelism)
- **--model_dir**: InternVL模型文件路径
- **--mem_fraction 0.8**: GPU显存使用比例
- **--trust_remote_code**: 允许加载自定义模型代码
- **--enable_multimodal**: 启用多模态功能

高级配置参数
------------

.. code-block:: bash

    --visual_infer_batch_size 2 \
    --cache_capacity 500 \
    --visual_dp dp_size \
    --visual_tp tp_size

- **--visual_infer_batch_size 2**: 视觉推理批处理大小
- **--cache_capacity 500**: 图像嵌入缓存容量
- **--visual_dp 2**: 视觉模型数据并行度
- **--visual_tp 2**: 视觉模型张量并行度

.. note:: 为了使每一个GPU的显存负载相同，需要visual_dp * visual_tp = tp，例如tp=2，则visual_dp=1, visual_tp=2。

ViT部署方式
-----------

ViT TP (张量并行)
^^^^^^^^^^^^^^^^^

- 默认使用
- --visual_tp tp_size 开启张量并行

ViT DP (数据并行)
^^^^^^^^^^^^^^^^^

- 将不同图像批次分布到多个GPU
- 每个GPU运行完整ViT模型副本
- --visual_dp dp_size 开启数据并行

图像缓存机制
------------
LightLLM 会对输入图片的embeddings进行缓存，多轮对话中，如果图片相同，则可以直接使用缓存的embeddings，避免重复推理。

- **--cache_capacity**: 控制缓存的image embed数量
- 根据图片MD5哈希值进行匹配
- 采用LRU(最近最少使用)淘汰机制
- 命中的图片cache可直接跳过ViT推理


测试
------------

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
