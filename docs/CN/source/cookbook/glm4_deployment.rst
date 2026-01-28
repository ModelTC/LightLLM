.. _glm4_deployment:

GLM-4.7-Flash 模型部署指南
===========================

LightLLM 支持 GLM-4.7-Flash (glm4_moe_lite) 模型系列的部署，该模型采用 MoE 架构。本文档提供详细的部署配置、函数调用和 MTP（多令牌预测）支持信息。

模型概述
--------

**主要特性：**

- 分组 MoE，支持 top-k 专家选择
- 支持 ``vanilla_with_att`` 和 ``eagle_with_att`` MTP 模式
- 兼容 FlashAttention3 后端
- 支持 XML 风格参数格式的函数调用

模型参考：https://huggingface.co/zai-org/GLM-4.7-Flash

推荐启动脚本 (H200)
-------------------

**基础启动命令：**

.. code-block:: bash

    LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 LOADWORKER=18 \
    python -m lightllm.server.api_server \
        --model_dir /path/to/GLM-4.7-Flash/ \
        --tp 1 \
        --max_req_total_len 202752 \
        --chunked_prefill_size 8192 \
        --llm_prefill_att_backend fa3 \
        --llm_decode_att_backend flashinfer \
        --graph_max_batch_size 512 \
        --tool_call_parser glm47 \
        --reasoning_parser glm45 \
        --host 0.0.0.0 \
        --port 8000

**参数说明：**

- ``LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1``: 启用 Triton 自动调优以获得最佳内核性能
- ``LOADWORKER=18``: 模型加载线程数，加快权重加载速度
- ``--tp 1``: 张量并行度（单 GPU）
- ``--max_req_total_len 202752``: 最大请求总长度
- ``--chunked_prefill_size 8192``: 预填充处理的分块大小
- ``--llm_prefill_att_backend fa3``: 预填充阶段使用 FlashAttention3
- ``--llm_decode_att_backend flashinfer``: 解码阶段使用 FlashInfer
- ``--graph_max_batch_size 512``: CUDA graph 最大批处理大小
- ``--tool_call_parser glm47``: 使用 GLM-4.7 函数调用解析器
- ``--reasoning_parser glm45``: 使用 GLM-4.5 推理解析器

MTP（多令牌预测）模式
---------------------

要启用 MTP 进行推测解码，请添加以下参数：

.. code-block:: bash

    LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 LOADWORKER=18 \
    python -m lightllm.server.api_server \
        --model_dir /path/to/GLM-4.7-Flash/ \
        --tp 1 \
        --max_req_total_len 202752 \
        --chunked_prefill_size 8192 \
        --llm_prefill_att_backend fa3 \
        --llm_decode_att_backend flashinfer \
        --graph_max_batch_size 512 \
        --tool_call_parser glm47 \
        --reasoning_parser glm45 \
        --mtp_step 4 \
        --mtp_mode eagle_with_att \
        --mtp_draft_model_dir /path/to/GLM-4.7-Flash/ \
        --host 0.0.0.0 \
        --port 8000

**MTP 参数说明：**

- ``--mtp_step 4``: 每个 MTP 步骤预测的令牌数
- ``--mtp_mode eagle_with_att``: MTP 模式（支持 ``vanilla_with_att`` 和 ``eagle_with_att``）
- ``--mtp_draft_model_dir``: MTP 草稿模型路径

函数调用支持
------------

GLM-4.7-Flash 使用新的 ``Glm47Detector`` 类来解析 XML 风格的工具调用。

**函数调用格式：**

.. code-block:: xml

    <tool_call>func_name
    <arg_key>key</arg_key><arg_value>value</arg_value>
    </tool_call>

**特性：**

- 完整的流式支持，支持增量解析
- 兼容 OpenAI 风格的函数调用 API

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
               "model": "GLM-4.7-Flash",
               "messages": [{"role": "user", "content": "你好"}],
               "max_tokens": 100
              }'

性能基准测试
------------

函数调用测试结果 (BFCL v3)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 20

   * - 类别
     - LightLLM
   * - simple
     - 62.50%
   * - multiple
     - 54.50%
   * - parallel
     - 69.50%
   * - parallel_multiple
     - 61.50%
   * - java
     - 66.00%
   * - javascript
     - 48.00%
   * - irrelevance
     - 83.33%
   * - live_simple
     - 45.74%
   * - live_multiple
     - 34.00%
   * - live_parallel
     - 25.00%
   * - live_parallel_multiple
     - 37.50%
   * - rest
     - 2.86%
   * - sql
     - 28.00%
   * - **总体**
     - **49.12%**

速度测试结果 (ShareGPT 2000 条提示，4×H200)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - 工作负载
     - 输出 (tok/s)
     - TTFT (ms)
     - 端到端延迟 (ms)
   * - burst
     - 6442
     - 11476
     - 27719
   * - high-conc (512)
     - **6728**
     - 1099
     - 11240
   * - moderate (10 req/s)
     - 1798
     - 196
     - 5746
   * - steady (5 req/s)
     - 917
     - 154
     - 2797

硬件要求
--------

**测试配置：**

- 4× NVIDIA H200 (每卡 80GB HBM3)
- NVLink 4.0 互联

**最低要求：**

- 基础部署需要单张 NVIDIA H100/H200 GPU（80GB 显存）
- 生产环境建议使用多 GPU 配置
