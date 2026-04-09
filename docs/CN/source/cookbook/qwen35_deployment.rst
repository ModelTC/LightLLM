.. _qwen35_deployment:

Qwen3.5 模型部署指南
=====================

LightLLM 支持 Qwen3.5 模型系列的部署。本指南以 `Qwen3.5-397B-A17B <https://huggingface.co/Qwen/Qwen3.5-397B-A17B>`_ 为例，介绍部署配置、思考/推理模式、多模态输入及推荐启动参数。

模型概述
--------

Qwen3.5-397B-A17B 是一个多模态混合专家模型，总参数量 397B，每个 token 激活 17B 参数。原生支持文本、图像和视频理解。

**主要特性：**

- **混合注意力架构**：60 层排列为 15 个重复组 ``[3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE)]``，交替使用线性注意力与全注意力（通过 ``full_attention_interval`` 控制）
- **稀疏 MoE**：共 512 个专家，每个 token 激活 10 个路由专家 + 1 个共享专家
- **原生多模态**：内置视觉编码器，支持图像和视频理解，无需单独的 "-VL" 变体
- **长上下文**：原生支持 262K 上下文，通过 YaRN 缩放可扩展至 1M+ tokens
- **多头旋转位置编码（MRoPE）**：交错旋转位置编码，``mrope_section=[11, 11, 10]``，用于空间/时间定位
- **思考/推理模式**：支持 ``qwen3`` 推理解析器，使用 ``<think>...</think>`` 标签（默认启用）

**已注册的模型类型：**

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

.. note::

    Qwen3.5 模型默认注册为多模态模型，多模态支持自动启用。若需纯文本部署，添加 ``--disable_vision`` 以跳过视觉编码器的加载，减少显存占用和启动时间。

推荐启动脚本
--------------

Qwen3.5-397B-A17B（8×H200）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 8 张 GPU 上部署完整的多模态 MoE 模型：

.. code-block:: bash

    LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 LOADWORKER=18 \
    python -m lightllm.server.api_server \
        --model_dir /path/to/Qwen3.5-397B-A17B/ \
        --tp 8 \
        --max_req_total_len 262144 \
        --chunked_prefill_size 8192 \
        --llm_prefill_att_backend fa3 \
        --llm_decode_att_backend flashinfer \
        --graph_max_batch_size 128 \
        --reasoning_parser qwen3 \
        --host 0.0.0.0 \
        --port 8000

**参数说明：**

- ``LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1``: 启用 Triton 自动调优以获得最佳内核性能
- ``LOADWORKER=18``: 模型加载线程数，加快权重加载速度
- ``--tp 8``: 8 卡张量并行（397B 参数模型必需）
- ``--max_req_total_len 262144``: 最大请求总长度，与模型原生 262K 上下文匹配
- ``--chunked_prefill_size 8192``: 预填充处理的分块大小，降低峰值显存占用
- ``--llm_prefill_att_backend fa3``: 预填充阶段使用 FlashAttention3（推荐 H200）
- ``--llm_decode_att_backend flashinfer``: 解码阶段使用 FlashInfer
- ``--graph_max_batch_size 128``: CUDA graph 最大批处理大小（显存不足时可减小）
- ``--reasoning_parser qwen3``: 启用 Qwen3 推理解析器，支持思考模式

纯文本模式（节省显存）
~~~~~~~~~~~~~~~~~~~~~~~

跳过视觉编码器加载以减少显存占用：

.. code-block:: bash

    LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 LOADWORKER=18 \
    python -m lightllm.server.api_server \
        --model_dir /path/to/Qwen3.5-397B-A17B/ \
        --tp 8 \
        --max_req_total_len 262144 \
        --chunked_prefill_size 8192 \
        --llm_prefill_att_backend fa3 \
        --llm_decode_att_backend flashinfer \
        --graph_max_batch_size 128 \
        --reasoning_parser qwen3 \
        --disable_vision \
        --host 0.0.0.0 \
        --port 8000

唯一区别是 ``--disable_vision``，阻止加载视觉编码器。此模式下模型仅接受文本输入。

思考/推理模式
-------------

Qwen3.5 默认启用思考模式。模型在生成最终答案之前，会在 ``<think>...</think>`` 标签内生成思维链推理过程。

**启用推理模式：**

在启动命令中添加 ``--reasoning_parser qwen3``（以上所有示例均已包含）。使用 OpenAI 兼容 API 时，在请求中设置 ``separate_reasoning: true`` 可单独获取思考内容：

.. code-block:: bash

    curl http://localhost:8000/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{
               "model": "Qwen3.5-397B-A17B",
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
               "model": "Qwen3.5-397B-A17B",
               "messages": [{"role": "user", "content": "你好"}],
               "max_tokens": 100,
               "enable_thinking": false
              }'

**推荐采样参数：**

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - 参数
     - 思考模式
     - 非思考模式
   * - temperature
     - 0.6
     - 0.7
   * - top_p
     - 0.95
     - 0.8
   * - top_k
     - 20
     - 20
   * - presence_penalty
     - 0.0
     - 1.5

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
               "model": "Qwen3.5-397B-A17B",
               "messages": [{"role": "user", "content": "你好"}],
               "max_tokens": 100,
               "temperature": 0.7,
               "top_p": 0.8,
               "enable_thinking": false
              }'

多模态测试（图像输入）
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    curl http://localhost:8000/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{
               "model": "Qwen3.5-397B-A17B",
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

**Qwen3.5-397B-A17B：**

- 总参数量 397B，每个 token 激活 17B（512 个专家，10 路由 + 1 共享）
- **最低要求**：8× NVIDIA H100/H200 GPU（每卡 80GB HBM），需 NVLink 互联
- 必须使用 ``--tp 8`` 以将模型权重分布到各 GPU
- 如遇到显存不足，可减小 ``--max_req_total_len`` 或 ``--graph_max_batch_size``
- 使用 ``--data_type fp8_e4m3`` 进行 FP8 KV 量化可进一步降低显存压力
