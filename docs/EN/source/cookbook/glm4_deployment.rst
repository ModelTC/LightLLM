.. _glm4_deployment:

GLM-4.7-Flash Model Deployment Guide
=====================================

LightLLM supports deployment of GLM-4.7-Flash (glm4_moe_lite) model family with MoE architecture. This document provides detailed information on deployment configuration, function calling, and MTP (Multi-Token Prediction) support.

Model Overview
--------------

**Key Features:**

- Grouped MoE with top-k expert selection
- Support for ``vanilla_with_att`` and ``eagle_with_att`` MTP modes
- Compatible with FlashAttention3 backend
- Function calling support with XML-style argument format

Model Reference: https://huggingface.co/zai-org/GLM-4.7-Flash

Recommended Launch Script (H200)
--------------------------------

**Basic Launch Command:**

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

**Parameter Description:**

- ``LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1``: Enable Triton autotuning for optimal kernel performance
- ``LOADWORKER=18``: Number of model loading threads for faster weight loading
- ``--tp 1``: Tensor parallelism degree (single GPU)
- ``--max_req_total_len 202752``: Maximum total request length
- ``--chunked_prefill_size 8192``: Chunk size for prefill processing
- ``--llm_prefill_att_backend fa3``: Use FlashAttention3 for prefill
- ``--llm_decode_att_backend flashinfer``: Use FlashInfer for decode
- ``--graph_max_batch_size 512``: Maximum batch size for CUDA graph
- ``--tool_call_parser glm47``: Use GLM-4.7 function calling parser
- ``--reasoning_parser glm45``: Use GLM-4.5 reasoning parser

MTP (Multi-Token Prediction) Mode
---------------------------------

To enable MTP for speculative decoding, add the following parameters:

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

**MTP Parameters:**

- ``--mtp_step 4``: Number of tokens to predict in each MTP step
- ``--mtp_mode eagle_with_att``: MTP mode (supports ``vanilla_with_att`` and ``eagle_with_att``)
- ``--mtp_draft_model_dir``: Path to the draft model for MTP

Function Calling Support
------------------------

GLM-4.7-Flash uses a new ``Glm47Detector`` class for parsing XML-style tool calls.

**Function Call Format:**

.. code-block:: xml

    <tool_call>func_name
    <arg_key>key</arg_key><arg_value>value</arg_value>
    </tool_call>

**Features:**

- Full streaming support for incremental parsing
- Compatible with OpenAI-style function calling API

Testing and Validation
----------------------

Basic Functionality Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    curl http://localhost:8000/generate \
         -H "Content-Type: application/json" \
         -d '{
               "inputs": "What is AI?",
               "parameters":{
                 "max_new_tokens": 100,
                 "frequency_penalty": 1
               }
              }'

OpenAI-Compatible Chat Completions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    curl http://localhost:8000/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{
               "model": "GLM-4.7-Flash",
               "messages": [{"role": "user", "content": "Hello"}],
               "max_tokens": 100
              }'

Performance Benchmarks
----------------------

Function Calling Test Results (BFCL v3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 20

   * - Category
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
   * - **OVERALL**
     - **49.12%**

Speed Test Results (ShareGPT 2000 prompts, 4×H200)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Workload
     - Output (tok/s)
     - TTFT (ms)
     - E2E Latency (ms)
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

Hardware Requirements
---------------------

**Tested Configuration:**

- 4× NVIDIA H200 (80GB HBM3 each)
- NVLink 4.0 interconnect

**Minimum Requirements:**

- Single NVIDIA H100/H200 GPU with 80GB memory for basic deployment
- Multiple GPUs recommended for production workloads
