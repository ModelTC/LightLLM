.. _qwen35_deployment:

Qwen3.5 Model Deployment Guide
===============================

LightLLM supports deployment of the Qwen3.5 model family. This guide uses `Qwen3.5-397B-A17B <https://huggingface.co/Qwen/Qwen3.5-397B-A17B>`_ as an example, covering deployment configuration, thinking/reasoning mode, multimodal input, and recommended launch parameters.

Model Overview
--------------

Qwen3.5-397B-A17B is a multimodal Mixture-of-Experts model with 397B total parameters and 17B active parameters per token. It natively supports text, image, and video understanding.

**Key Features:**

- **Hybrid Attention Architecture**: 60 layers arranged as 15 repeating groups of ``[3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE)]``, alternating between linear attention and full attention (controlled by ``full_attention_interval``)
- **Sparse MoE**: 512 total experts, 10 routed + 1 shared expert activated per token
- **Native Multimodal**: Built-in vision encoder for image and video understanding — no separate "-VL" variant needed
- **Long Context**: 262K native context, extensible to 1M+ tokens with YaRN scaling
- **Multi-head RoPE (MRoPE)**: Interleaved rotary position embeddings with ``mrope_section=[11, 11, 10]`` for spatial/temporal positioning
- **Thinking/Reasoning Mode**: Supports ``qwen3`` reasoning parser with ``<think>...</think>`` tags (enabled by default)

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

Qwen3.5-397B-A17B (8×H200)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deploy the full multimodal MoE model on 8 GPUs:

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

**Parameter Description:**

- ``LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1``: Enable Triton autotuning for optimal kernel performance
- ``LOADWORKER=18``: Number of model loading threads for faster weight loading
- ``--tp 8``: Tensor parallelism across 8 GPUs (required for 397B parameter model)
- ``--max_req_total_len 262144``: Maximum total request length matching the model's native 262K context
- ``--chunked_prefill_size 8192``: Chunk size for prefill processing, reduces peak memory usage
- ``--llm_prefill_att_backend fa3``: Use FlashAttention3 for prefill (recommended for H200)
- ``--llm_decode_att_backend flashinfer``: Use FlashInfer for decode phase
- ``--graph_max_batch_size 128``: Maximum batch size for CUDA graph optimization (reduce if OOM)
- ``--reasoning_parser qwen3``: Enable Qwen3 reasoning parser for thinking mode

Text-only Mode (Save Memory)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To skip loading the vision encoder and reduce memory usage:

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

The only difference is ``--disable_vision``, which prevents the vision encoder from being loaded. The model will only accept text input in this mode.

Thinking/Reasoning Mode
-----------------------

Qwen3.5 has thinking mode enabled by default. The model generates chain-of-thought reasoning inside ``<think>...</think>`` tags before producing the final answer.

**Enabling Reasoning Mode:**

Add ``--reasoning_parser qwen3`` to your launch command (included in all examples above). When using the OpenAI-compatible API, set ``separate_reasoning: true`` in the request to receive thinking content separately:

.. code-block:: bash

    curl http://localhost:8000/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{
               "model": "Qwen3.5-397B-A17B",
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
               "model": "Qwen3.5-397B-A17B",
               "messages": [{"role": "user", "content": "Hello"}],
               "max_tokens": 100,
               "enable_thinking": false
              }'

**Recommended Sampling Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Parameter
     - Thinking Mode
     - Non-Thinking Mode
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
               "model": "Qwen3.5-397B-A17B",
               "messages": [{"role": "user", "content": "Hello"}],
               "max_tokens": 100,
               "temperature": 0.7,
               "top_p": 0.8,
               "enable_thinking": false
              }'

Multimodal Testing (Image Input)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
                     {"type": "text", "text": "Describe this image."}
                   ]
                 }
               ],
               "max_tokens": 200
              }'

Hardware Requirements
---------------------

**Qwen3.5-397B-A17B:**

- 397B total parameters, 17B activated per token (512 experts, 10 routed + 1 shared)
- **Minimum**: 8× NVIDIA H100/H200 GPUs (80GB HBM each) with NVLink interconnect
- ``--tp 8`` required to fit model weights across GPUs
- Reduce ``--max_req_total_len`` or ``--graph_max_batch_size`` if encountering OOM errors
- Use ``--data_type fp8_e4m3`` for FP8 KV quantization to further reduce memory pressure
- For multimodal OOMs driven by dynamic-resolution images or video, cap the
  per-step ViT workload with ``--visual_batch_max_tokens`` (e.g. ``16384``).
  This bounds peak ViT memory the same way ``--batch_max_tokens`` bounds the
  LLM prefill. Setting this alone also derives a default single-image cap
  (``--visual_image_max_tokens`` is implicitly set to the same value), which
  rejects any single image that couldn't fit in one batch — closing the "first
  image always admitted" deadlock-avoidance hole. Override
  ``--visual_image_max_tokens`` separately only if you need a stricter single-
  image limit.
