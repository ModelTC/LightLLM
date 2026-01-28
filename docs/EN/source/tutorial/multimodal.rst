Multimodal Model Launch Configuration
=====================================

LightLLM supports inference for various multimodal models. Below, using InternVL as an example, we explain the launch commands for multimodal services.

Basic Launch Command
--------------------

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

Core Parameter Description
--------------------------

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

- **INTERNVL_IMAGE_LENGTH**: Set the image token length for InternVL model, default is 256
- **LOADWORKER**: Set the number of worker processes for model loading

Basic Service Parameters
^^^^^^^^^^^^^^^^^^^^^^^^

- **--port 8080**: API server listening port
- **--tp 2**: Tensor parallelism degree
- **--model_dir**: InternVL model file path
- **--mem_fraction 0.8**: GPU memory usage ratio
- **--trust_remote_code**: Allow loading custom model code
- **--enable_multimodal**: Enable multimodal functionality

Advanced Configuration Parameters
---------------------------------

.. code-block:: bash

    --visual_infer_batch_size 2 \
    --cache_capacity 500 \
    --visual_dp dp_size \
    --visual_tp tp_size

- **--visual_infer_batch_size 2**: Visual inference batch size
- **--cache_capacity 500**: Image embedding cache capacity
- **--visual_dp 2**: Visual model data parallelism degree
- **--visual_tp 2**: Visual model tensor parallelism degree

.. note:: To ensure equal memory load on each GPU, visual_dp * visual_tp = tp is required. For example, if tp=2, then visual_dp=1, visual_tp=2.

ViT Deployment Methods
----------------------

ViT TP (Tensor Parallel)
^^^^^^^^^^^^^^^^^^^^^^^^^

- Default usage
- --visual_tp tp_size enables tensor parallelism

ViT DP (Data Parallel)
^^^^^^^^^^^^^^^^^^^^^^^

- Distribute different image batches to multiple GPUs
- Each GPU runs a complete ViT model copy
- --visual_dp dp_size enables data parallelism

Image Caching Mechanism
-----------------------
LightLLM caches embeddings of input images. In multi-turn conversations, if the images are the same, cached embeddings can be used directly, avoiding repeated inference.

- **--cache_capacity**: Controls the number of cached image embeds
- Matching based on image MD5 hash value
- Uses LRU (Least Recently Used) eviction mechanism
- Hit image cache can directly skip ViT inference

Testing
-------

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
    What is this?<|im_end|>
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