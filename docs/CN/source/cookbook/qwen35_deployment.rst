.. _qwen35_deployment:

Qwen3.5 模型部署指南
=====================

LightLLM 支持 Qwen3.5 模型系列的部署，包括稠密型（``qwen3_5``）和混合专家型（``qwen3_5_moe``）两种变体。本文档提供纯文本和多模态（视觉）模式的详细部署配置、思考/推理模式支持以及推荐的启动参数。

模型概述
--------

Qwen3.5 是新一代大语言模型，采用混合注意力架构，并可选支持多模态能力。

**主要特性：**

- **混合注意力架构**：交替使用全注意力和门控 Delta 网络（线性注意力），通过 ``full_attention_interval`` 控制
- **多模态支持**：通过视觉编码器实现图像和视频理解（继承自 Qwen3VL）
- **稠密和 MoE 变体**：``qwen3_5`` 为稠密 MLP，``qwen3_5_moe`` 为混合专家模型
- **多头旋转位置编码（MRoPE）**：针对多模态空间/时间定位优化的交错旋转位置编码
- **思考/推理模式**：支持 ``qwen3`` 推理解析器，使用 ``<think>...</think>`` 标签进行思维链生成

**支持的模型类型：**

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - 模型类型
     - 架构
     - 说明
   * - ``qwen3_5``
     - 稠密 + 多模态
     - 稠密 MLP，带视觉编码器
   * - ``qwen3_5_moe``
     - MoE + 多模态
     - 混合专家模型，带视觉编码器
   * - ``qwen3_5_text``
     - 稠密 + 纯文本
     - 稠密 MLP，无视觉编码器
   * - ``qwen3_5_moe_text``
     - MoE + 纯文本
     - 混合专家模型，无视觉编码器

推荐启动脚本
--------------

纯文本稠密模型
~~~~~~~~~~~~~~~

在单 GPU 上部署稠密纯文本变体：

.. code-block:: bash

    LOADWORKER=18 \
    python -m lightllm.server.api_server \
        --model_dir /path/to/Qwen3.5/ \
        --tp 1 \
        --max_req_total_len 32768 \
        --chunked_prefill_size 8192 \
        --graph_max_batch_size 256 \
        --reasoning_parser qwen3 \
        --host 0.0.0.0 \
        --port 8000

**参数说明：**

- ``LOADWORKER=18``: 模型加载线程数，加快权重加载速度
- ``--tp 1``: 张量并行度（小模型使用单 GPU；大模型需增加）
- ``--max_req_total_len 32768``: 最大请求总长度（输入 + 输出 token 数）
- ``--chunked_prefill_size 8192``: 预填充处理的分块大小，降低峰值显存占用
- ``--graph_max_batch_size 256``: CUDA graph 最大批处理大小
- ``--reasoning_parser qwen3``: 启用 Qwen3 推理解析器，支持思考模式

多模态稠密模型（带视觉能力）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

部署支持图像/视频理解的多模态变体：

.. code-block:: bash

    LOADWORKER=18 \
    python -m lightllm.server.api_server \
        --model_dir /path/to/Qwen3.5-VL/ \
        --tp 2 \
        --max_req_total_len 32768 \
        --chunked_prefill_size 8192 \
        --graph_max_batch_size 256 \
        --reasoning_parser qwen3 \
        --enable_multimodal \
        --host 0.0.0.0 \
        --port 8000

**额外参数：**

- ``--enable_multimodal``: 启用多模态（图像/视频）输入处理
- ``--tp 2``: 多模态模型通常较大，建议使用多 GPU 张量并行

MoE 模型
~~~~~~~~~

使用多 GPU 部署 MoE 变体：

.. code-block:: bash

    LOADWORKER=18 \
    python -m lightllm.server.api_server \
        --model_dir /path/to/Qwen3.5-MoE/ \
        --tp 4 \
        --max_req_total_len 32768 \
        --chunked_prefill_size 8192 \
        --graph_max_batch_size 128 \
        --reasoning_parser qwen3 \
        --host 0.0.0.0 \
        --port 8000

.. note::

    MoE 模型总参数量更大，但每个 token 仅激活部分专家。建议使用较高的张量并行度（``--tp 4`` 或 ``--tp 8``）以将专家权重分布到多个 GPU。如遇到显存不足，可减小 ``--graph_max_batch_size``。

高性能启动（H200）
~~~~~~~~~~~~~~~~~~~

在 H200 GPU 上使用 FlashAttention3 获得最佳性能：

.. code-block:: bash

    LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 LOADWORKER=18 \
    python -m lightllm.server.api_server \
        --model_dir /path/to/Qwen3.5/ \
        --tp 1 \
        --max_req_total_len 32768 \
        --chunked_prefill_size 8192 \
        --llm_prefill_att_backend fa3 \
        --llm_decode_att_backend flashinfer \
        --graph_max_batch_size 256 \
        --reasoning_parser qwen3 \
        --host 0.0.0.0 \
        --port 8000

**性能调优参数：**

- ``LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1``: 启用 Triton 自动调优以获得最佳内核性能
- ``--llm_prefill_att_backend fa3``: 预填充阶段使用 FlashAttention3（推荐 H200）
- ``--llm_decode_att_backend flashinfer``: 解码阶段使用 FlashInfer

思考/推理模式
-------------

Qwen3.5 支持思考模式，模型在生成最终答案之前，会在 ``<think>...</think>`` 标签内生成思维链推理过程。

**启用推理模式：**

在启动命令中添加 ``--reasoning_parser qwen3``（以上所有示例均已包含）。使用 OpenAI 兼容 API 时，在请求中设置 ``separate_reasoning: true`` 可单独获取思考内容：

.. code-block:: bash

    curl http://localhost:8000/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{
               "model": "Qwen3.5",
               "messages": [{"role": "user", "content": "请逐步求解：23 * 47 等于多少？"}],
               "max_tokens": 500,
               "separate_reasoning": true
              }'

响应中将包含 ``reasoning_content`` 字段（模型思考过程）和 ``content`` 字段（最终答案）。

**针对特定请求禁用思考：**

若需要更快的响应速度，可在请求中设置 ``enable_thinking: false`` 以使用非思考模式：

.. code-block:: bash

    curl http://localhost:8000/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{
               "model": "Qwen3.5",
               "messages": [{"role": "user", "content": "你好"}],
               "max_tokens": 100,
               "enable_thinking": false
              }'

FP8 KV 缓存量化
-----------------

Qwen3.5 支持 FP8 KV 缓存量化以减少显存占用。创建校准配置：

.. code-block:: json

    {
        "kv_quant_type": "fp8_e4m3"
    }

然后在启动命令中添加以下参数：

.. code-block:: bash

    --data_type fp8_e4m3

这可以显著减少 KV 缓存的显存占用，从而支持更大的批处理大小和更长的序列。

测试与验证
----------

基础功能测试
~~~~~~~~~~~~

.. code-block:: bash

    curl http://localhost:8000/generate \
         -H "Content-Type: application/json" \
         -d '{
               "inputs": "什么是人工智能？",
               "parameters":{
                 "max_new_tokens": 100,
                 "frequency_penalty": 1
               }
              }'

OpenAI 兼容聊天接口
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    curl http://localhost:8000/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{
               "model": "Qwen3.5",
               "messages": [{"role": "user", "content": "你好"}],
               "max_tokens": 100
              }'

多模态测试（图像输入）
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    curl http://localhost:8000/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{
               "model": "Qwen3.5",
               "messages": [
                 {
                   "role": "user",
                   "content": [
                     {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                     {"type": "text", "text": "请描述这张图片。"}
                   ]
                 }
               ],
               "max_tokens": 200
              }'

硬件要求
--------

**推荐配置：**

.. list-table::
   :header-rows: 1
   :widths: 30 20 25 25

   * - 变体
     - TP
     - GPU 显存
     - 推荐 GPU
   * - 稠密型（小）
     - 1
     - 80GB+
     - 1× H100/H200
   * - 稠密型（大）
     - 2-4
     - 每卡 80GB+
     - 2-4× H100/H200
   * - MoE
     - 4-8
     - 每卡 80GB+
     - 4-8× H100/H200
   * - 多模态
     - 2+
     - 每卡 80GB+
     - 2+× H100/H200

.. note::

    实际 GPU 显存需求取决于模型大小、序列长度和批处理大小。可通过 ``--graph_max_batch_size`` 和 ``--max_req_total_len`` 控制显存占用。MoE 模型总参数量更大，但每个 token 仅激活部分参数，因此显存需求主要取决于专家总数。
