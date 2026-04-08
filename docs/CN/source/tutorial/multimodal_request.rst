多模态请求指南
============================

LightLLM 提供两种 API 格式来接收多模态请求：OpenAI 兼容格式（推荐）和 Legacy 格式。本文档介绍如何向 LightLLM 服务发送包含图片和音频的多模态请求。

OpenAI 兼容格式（推荐）
------------------------

使用 ``POST /v1/chat/completions`` 端点，与 OpenAI API 格式完全兼容。

图片输入方式
^^^^^^^^^^^^

LightLLM 支持三种图片输入方式：

**1. URL 方式**

.. code-block:: json

    {
        "type": "image_url",
        "image_url": {
            "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"
        }
    }

**2. Base64 编码**

.. code-block:: json

    {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
        }
    }

**3. 本地文件路径**

.. code-block:: json

    {
        "type": "image_url",
        "image_url": {
            "url": "file:///path/to/image.jpg"
        }
    }

音频输入方式
^^^^^^^^^^^^

音频同样支持 URL 和 Base64 两种方式：

.. code-block:: json

    {
        "type": "audio_url",
        "audio_url": {
            "url": "https://example.com/audio.wav"
        }
    }

使用 curl 发送请求
^^^^^^^^^^^^^^^^^^

**基本图片请求**

.. code-block:: bash

    curl http://127.0.0.1:8080/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "qwen",
        "messages": [
          {
            "role": "user",
            "content": [
              {
                "type": "image_url",
                "image_url": {
                  "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"
                }
              },
              {
                "type": "text",
                "text": "请描述这张图片的内容。"
              }
            ]
          }
        ],
        "max_completion_tokens": 512,
        "temperature": 0.7
      }'

**流式输出**

.. code-block:: bash

    curl http://127.0.0.1:8080/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "qwen",
        "messages": [
          {
            "role": "user",
            "content": [
              {
                "type": "image_url",
                "image_url": {
                  "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"
                }
              },
              {
                "type": "text",
                "text": "请描述这张图片的内容。"
              }
            ]
          }
        ],
        "max_completion_tokens": 512,
        "stream": true
      }'

使用 Python requests 发送请求
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import requests

    url = "http://127.0.0.1:8080/v1/chat/completions"
    image_url = "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"

    payload = {
        "model": "qwen",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "请描述这张图片的内容。"},
                ],
            }
        ],
        "max_completion_tokens": 512,
        "temperature": 0.7,
    }

    response = requests.post(url, json=payload)
    print(response.json())

使用 OpenAI SDK 发送请求
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from openai import OpenAI

    client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")

    image_url = "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"

    # 非流式
    response = client.chat.completions.create(
        model="qwen",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "请描述这张图片的内容。"},
                ],
            }
        ],
        max_completion_tokens=512,
        temperature=0.7,
    )
    print(response.choices[0].message.content)

    # 流式
    stream = client.chat.completions.create(
        model="qwen",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "请描述这张图片的内容。"},
                ],
            }
        ],
        max_completion_tokens=512,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

多图片请求
^^^^^^^^^^

在 ``content`` 列表中传入多个 ``image_url`` 即可：

.. code-block:: python

    response = client.chat.completions.create(
        model="qwen",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}},
                    {"type": "text", "text": "请比较这两张图片的区别。"},
                ],
            }
        ],
        max_completion_tokens=512,
    )

多轮对话
^^^^^^^^

多轮对话中，相同图片会自动命中 embed 缓存，无需重复推理：

.. code-block:: python

    image_url = "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "这张图片里有什么？"},
            ],
        },
    ]

    # 第一轮
    response = client.chat.completions.create(
        model="qwen", messages=messages, max_completion_tokens=512
    )
    answer = response.choices[0].message.content
    messages.append({"role": "assistant", "content": answer})

    # 第二轮（同一张图片会命中缓存）
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": "请更详细地描述图片中的文字内容。"},
        ],
    })
    response = client.chat.completions.create(
        model="qwen", messages=messages, max_completion_tokens=512
    )

Legacy 格式
------------

使用 ``POST /generate`` 端点，需要手动构造 prompt 模板和 multimodal_params。

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
                with open(uri, "rb") as fin:
                    b64 = base64.b64encode(fin.read()).decode("utf-8")
                images.append({"type": "base64", "data": b64})

        data = {
            "inputs": query,
            "parameters": {
                "max_new_tokens": 512,
                "do_sample": False,
            },
            "multimodal_params": {
                "images": images,
            },
        }

        url = "http://127.0.0.1:8080/generate"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, data=json.dumps(data))
        return response


    response = run(
        uris=["https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"],
        query="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n<img></img>\n请描述这张图片的内容。<|im_end|>\n"
              "<|im_start|>assistant\n",
    )

    if response.status_code == 200:
        print(f"Result: {response.json()}")
    else:
        print(f"Error: {response.status_code}, {response.text}")

.. note:: Legacy 格式需要手动构造 chat template，推荐使用 OpenAI 兼容格式以避免模板错误。

常用采样参数
------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - 参数
     - 类型
     - 默认值
     - 说明
   * - max_completion_tokens
     - int
     - 16384
     - 最大输出 token 数
   * - temperature
     - float
     - 1.0
     - 采样温度，越高越随机
   * - top_p
     - float
     - 1.0
     - 核采样参数
   * - top_k
     - int
     - -1
     - Top-K 采样，-1 表示禁用
   * - do_sample
     - bool
     - true
     - 是否启用采样（false 为贪心解码）
   * - repetition_penalty
     - float
     - 1.0
     - 重复惩罚系数，>1 抑制重复
   * - stream
     - bool
     - false
     - 是否流式输出
   * - stop
     - str/list
     - -
     - 停止生成的序列
   * - seed
     - int
     - -1
     - 随机种子，-1 为随机

响应格式
--------

**非流式响应**

.. code-block:: json

    {
        "id": "chatcmpl-xxx",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "qwen",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "这张图片显示的是..."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 256,
            "completion_tokens": 128,
            "total_tokens": 384
        }
    }

**流式响应**

每个 chunk 以 ``data: `` 前缀的 SSE 格式返回：

.. code-block:: json

    {
        "id": "chatcmpl-xxx",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": "这张"
                },
                "finish_reason": null
            }
        ]
    }

环境变量
--------

- **REQUEST_TIMEOUT**: 下载远程图片/音频的超时时间（秒），默认 5
- **REQUEST_PROXY**: 下载远程资源时使用的代理地址
