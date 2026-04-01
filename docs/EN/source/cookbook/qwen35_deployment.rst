.. _qwen35_deployment:

Qwen3.5 Model Deployment Guide
===============================

LightLLM supports deployment of Qwen3.5 model family, including both dense (``qwen3_5``) and MoE (``qwen3_5_moe``) variants. This document provides detailed information on deployment configuration for text-only and multimodal (vision) modes, thinking/reasoning support, and recommended launch parameters.

Model Overview
--------------

Qwen3.5 is a next-generation large language model featuring a hybrid attention architecture and optional multimodal capabilities.

**Key Features:**

- **Hybrid Attention Architecture**: Alternates between full attention and Gated Delta Networks (linear attention), controlled by ``full_attention_interval``
- **Multimodal Support**: Image and video understanding via vision encoder (inherited from Qwen3VL)
- **Dense and MoE Variants**: ``qwen3_5`` for dense MLP, ``qwen3_5_moe`` for Mixture-of-Experts
- **Multi-head RoPE (MRoPE)**: Interleaved rotary position embeddings optimized for multimodal spatial/temporal positioning
- **Thinking/Reasoning Mode**: Supports ``qwen3`` reasoning parser with ``<think>...</think>`` tags for chain-of-thought generation

**Registered Model Types:**

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Model Type
     - Architecture
     - Description
   * - ``qwen3_5``
     - Dense + Multimodal
     - Dense MLP with vision encoder
   * - ``qwen3_5_moe``
     - MoE + Multimodal
     - Mixture-of-Experts with vision encoder

.. note::

    Qwen3.5 models are registered as multimodal by default. Multimodal support is automatically enabled unless explicitly disabled. For text-only deployment, add ``--disable_vision`` to skip loading the vision encoder, which reduces memory usage and startup time.

Recommended Launch Scripts
--------------------------

Text-only Dense Model
~~~~~~~~~~~~~~~~~~~~~

For deploying the dense text-only variant on a single GPU:

.. code-block:: bash

    LOADWORKER=18 \
    python -m lightllm.server.api_server \
        --model_dir /path/to/Qwen3.5/ \
        --tp 1 \
        --max_req_total_len 32768 \
        --chunked_prefill_size 8192 \
        --graph_max_batch_size 256 \
        --reasoning_parser qwen3 \
        --disable_vision \
        --host 0.0.0.0 \
        --port 8000

**Parameter Description:**

- ``LOADWORKER=18``: Number of model loading threads for faster weight loading
- ``--tp 1``: Tensor parallelism degree (single GPU for small models; increase for larger variants)
- ``--max_req_total_len 32768``: Maximum total request length (input + output tokens)
- ``--chunked_prefill_size 8192``: Chunk size for prefill processing, reduces peak memory usage
- ``--graph_max_batch_size 256``: Maximum batch size for CUDA graph optimization
- ``--reasoning_parser qwen3``: Enable Qwen3 reasoning parser for thinking mode support
- ``--disable_vision``: Disable the vision encoder for text-only deployment, saving memory

Multimodal Dense Model (with Vision)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For deploying the multimodal variant with image/video understanding:

.. code-block:: bash

    LOADWORKER=18 \
    python -m lightllm.server.api_server \
        --model_dir /path/to/Qwen3.5-VL/ \
        --tp 2 \
        --max_req_total_len 32768 \
        --chunked_prefill_size 8192 \
        --graph_max_batch_size 256 \
        --reasoning_parser qwen3 \
        --host 0.0.0.0 \
        --port 8000

**Additional Parameters:**

- ``--tp 2``: Multimodal models are typically larger and benefit from multi-GPU tensor parallelism

.. note::

    Multimodal support is enabled by default for Qwen3.5 models. No additional flag is needed to enable vision capabilities.

MoE Model
~~~~~~~~~~

For deploying the MoE variant with multi-GPU support:

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

    MoE models have more parameters but only activate a subset per token. Use higher tensor parallelism (``--tp 4`` or ``--tp 8``) to distribute expert weights across GPUs. Reduce ``--graph_max_batch_size`` if encountering OOM errors.

High-Performance Launch (H200)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For optimal performance on H200 GPUs with FlashAttention3:

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

**Performance Tuning Parameters:**

- ``LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1``: Enable Triton autotuning for optimal kernel performance
- ``--llm_prefill_att_backend fa3``: Use FlashAttention3 for prefill (H200 recommended)
- ``--llm_decode_att_backend flashinfer``: Use FlashInfer for decode phase

Thinking/Reasoning Mode
-----------------------

Qwen3.5 supports a thinking mode where the model generates chain-of-thought reasoning inside ``<think>...</think>`` tags before producing the final answer.

**Enabling Reasoning Mode:**

Add ``--reasoning_parser qwen3`` to your launch command (included in all examples above). When using the OpenAI-compatible API, set ``separate_reasoning: true`` in the request to receive thinking content separately:

.. code-block:: bash

    curl http://localhost:8000/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{
               "model": "Qwen3.5",
               "messages": [{"role": "user", "content": "Solve step by step: what is 23 * 47?"}],
               "max_tokens": 500,
               "separate_reasoning": true
              }'

The response will include a ``reasoning_content`` field with the model's thinking process and a ``content`` field with the final answer.

**Disabling Thinking for Specific Requests:**

To use the model in non-thinking mode for faster responses, set ``enable_thinking: false`` in the request:

.. code-block:: bash

    curl http://localhost:8000/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{
               "model": "Qwen3.5",
               "messages": [{"role": "user", "content": "Hello"}],
               "max_tokens": 100,
               "enable_thinking": false
              }'

FP8 KV Cache Quantization
--------------------------

Qwen3.5 supports FP8 KV cache quantization to reduce memory usage. Create a calibration config:

.. code-block:: json

    {
        "kv_quant_type": "fp8_e4m3"
    }

Then add the following parameter to your launch command:

.. code-block:: bash

    --data_type fp8_e4m3

This can significantly reduce KV cache memory usage, allowing larger batch sizes and longer sequences.

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    curl http://localhost:8000/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{
               "model": "Qwen3.5",
               "messages": [{"role": "user", "content": "Hello"}],
               "max_tokens": 100
              }'

Multimodal Testing (Image Input)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
                     {"type": "text", "text": "Describe this image."}
                   ]
                 }
               ],
               "max_tokens": 200
              }'

Hardware Requirements
---------------------

**Recommended Configurations:**

.. list-table::
   :header-rows: 1
   :widths: 30 20 25 25

   * - Variant
     - TP
     - GPU Memory
     - Recommended GPUs
   * - Dense (small)
     - 1
     - 80GB+
     - 1× H100/H200
   * - Dense (large)
     - 2-4
     - 80GB+ per GPU
     - 2-4× H100/H200
   * - MoE
     - 4-8
     - 80GB+ per GPU
     - 4-8× H100/H200
   * - Multimodal
     - 2+
     - 80GB+ per GPU
     - 2+× H100/H200

.. note::

    Actual GPU memory requirements depend on model size, sequence length, and batch size. Use ``--graph_max_batch_size`` and ``--max_req_total_len`` to control memory usage. MoE models have higher total parameter counts but activate fewer parameters per token, so memory requirements scale with total expert count.
