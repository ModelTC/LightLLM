Multimodal Request Guide
============================

LightLLM provides two API formats for multimodal requests: OpenAI-compatible format (recommended) and Legacy format. This guide shows how to send requests with images and audio to a LightLLM server.

OpenAI-Compatible Format (Recommended)
---------------------------------------

Use the ``POST /v1/chat/completions`` endpoint, fully compatible with the OpenAI API.

Image Input Methods
^^^^^^^^^^^^^^^^^^^

LightLLM supports three image input methods:

**1. URL**

.. code-block:: json

    {
        "type": "image_url",
        "image_url": {
            "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"
        }
    }

**2. Base64 Encoded**

.. code-block:: json

    {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
        }
    }

**3. Local File Path**

.. code-block:: json

    {
        "type": "image_url",
        "image_url": {
            "url": "file:///path/to/image.jpg"
        }
    }

Audio Input Methods
^^^^^^^^^^^^^^^^^^^

Audio supports both URL and Base64 methods:

.. code-block:: json

    {
        "type": "audio_url",
        "audio_url": {
            "url": "https://example.com/audio.wav"
        }
    }

Using curl
^^^^^^^^^^

**Basic Image Request**

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
                "text": "Describe the content of this image."
              }
            ]
          }
        ],
        "max_completion_tokens": 512,
        "temperature": 0.7
      }'

**Streaming Output**

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
                "text": "Describe the content of this image."
              }
            ]
          }
        ],
        "max_completion_tokens": 512,
        "stream": true
      }'

Using Python requests
^^^^^^^^^^^^^^^^^^^^^

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
                    {"type": "text", "text": "Describe the content of this image."},
                ],
            }
        ],
        "max_completion_tokens": 512,
        "temperature": 0.7,
    }

    response = requests.post(url, json=payload)
    print(response.json())

Using OpenAI SDK
^^^^^^^^^^^^^^^^

.. code-block:: python

    from openai import OpenAI

    client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")

    image_url = "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"

    # Non-streaming
    response = client.chat.completions.create(
        model="qwen",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Describe the content of this image."},
                ],
            }
        ],
        max_completion_tokens=512,
        temperature=0.7,
    )
    print(response.choices[0].message.content)

    # Streaming
    stream = client.chat.completions.create(
        model="qwen",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Describe the content of this image."},
                ],
            }
        ],
        max_completion_tokens=512,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

Multi-Image Request
^^^^^^^^^^^^^^^^^^^

Pass multiple ``image_url`` items in the ``content`` list:

.. code-block:: python

    response = client.chat.completions.create(
        model="qwen",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}},
                    {"type": "text", "text": "Compare the differences between these two images."},
                ],
            }
        ],
        max_completion_tokens=512,
    )

Multi-Turn Conversation
^^^^^^^^^^^^^^^^^^^^^^^

In multi-turn conversations, identical images automatically hit the embed cache, avoiding redundant inference:

.. code-block:: python

    image_url = "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "What is in this image?"},
            ],
        },
    ]

    # First turn
    response = client.chat.completions.create(
        model="qwen", messages=messages, max_completion_tokens=512
    )
    answer = response.choices[0].message.content
    messages.append({"role": "assistant", "content": answer})

    # Second turn (same image hits cache)
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": "Describe the text content in the image in more detail."},
        ],
    })
    response = client.chat.completions.create(
        model="qwen", messages=messages, max_completion_tokens=512
    )

Legacy Format
--------------

Use the ``POST /generate`` endpoint. Requires manually constructing prompt templates and multimodal_params.

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
              "<|im_start|>user\n<img></img>\nDescribe the content of this image.<|im_end|>\n"
              "<|im_start|>assistant\n",
    )

    if response.status_code == 200:
        print(f"Result: {response.json()}")
    else:
        print(f"Error: {response.status_code}, {response.text}")

.. note:: The Legacy format requires manually constructing chat templates. The OpenAI-compatible format is recommended to avoid template errors.

Common Sampling Parameters
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - max_completion_tokens
     - int
     - 16384
     - Maximum output tokens
   * - temperature
     - float
     - 1.0
     - Sampling temperature, higher is more random
   * - top_p
     - float
     - 1.0
     - Nucleus sampling parameter
   * - top_k
     - int
     - -1
     - Top-K sampling, -1 to disable
   * - do_sample
     - bool
     - true
     - Enable sampling (false for greedy decoding)
   * - repetition_penalty
     - float
     - 1.0
     - Repetition penalty, >1 suppresses repetition
   * - stream
     - bool
     - false
     - Enable streaming output
   * - stop
     - str/list
     - -
     - Stop generation sequences
   * - seed
     - int
     - -1
     - Random seed, -1 for random

Response Format
----------------

**Non-Streaming Response**

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
                    "content": "The image shows..."
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

**Streaming Response**

Each chunk is returned in SSE format with the ``data: `` prefix:

.. code-block:: json

    {
        "id": "chatcmpl-xxx",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": "The image"
                },
                "finish_reason": null
            }
        ]
    }

Environment Variables
---------------------

- **REQUEST_TIMEOUT**: Timeout (seconds) for downloading remote images/audio, default 5
- **REQUEST_PROXY**: Proxy address for downloading remote resources
