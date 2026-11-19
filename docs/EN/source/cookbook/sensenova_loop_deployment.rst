.. _sensenova_loop_deployment:

SenseNova Loop Deployment Guide
==============================

Start the Server
----------------

This example starts a BF16 service on four GPUs. Run it from the repository root:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0,1,2,3 LOADWORKER=4 LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 \
    python -m lightllm.server.api_server \
        --model_dir /path/to/SenseNova-Loop \
        --model_name sensenova_loop \
        --tp 4 \
        --port 8000

Single RTX 4090 / 4090D (24 GiB)
--------------------------------

For a single request, use a Loop AWQ export with INT4 embedding and lm_head,
BF16 vision weights, and FP8 KV. The following recipe keeps decode CUDA Graph
enabled and runs ViT in the LLM worker to share its CUDA context and allocator.
It uses native LightLLM inference, scheduling and paged KV storage.

The runtime-scale export is identified by ``config.json``'s ``loop_awq_config``:
``bits=4``, ``group_size=128``, ``zero_point=true``,
``linear_layout="awq_gemm"``, ``input_scale_mode="runtime"`` and
``embedding_layout="row_major_int4"``. It contains ``unquantized.safetensors``
and ``modules/*.safetensors``. Keep the tokenizer, chat template and image
processor files in the model directory. Ordinary folded AWQ exports continue
to use the existing AWQ loader; they are a different checkpoint format.

Install a compatible vLLM build for AWQ-Marlin and FlashInfer for Ada FP8 KV.
Provide a **per_tensor** FP8 calibration file generated for this checkpoint and
its loop execution count; see :doc:`../tutorial/fp8_kv_quantization`.
The calibration rows follow the full-attention execution order: for a 32-layer
model with layers [4, 28) executed twice, there are 14 independent cache rows.
A calibration file from a different loop count or another model is unsuitable.

Run from the repository root:

.. code-block:: bash

    MODEL_DIR=/path/to/SenseNova-Loop-AWQ \
    KV_CALIB=/path/to/kv_cache_calib_per_tensor.json \
    CUDA_VISIBLE_DEVICES=0 CONTEXT_LENGTH=262144 PORT=8000 \
    bash test/start_scripts/sensenova_loop/sensenova_loop_4090.sh

Use ``DRY_RUN=1`` to print the exact command. Set ``CONTEXT_LENGTH`` to ``65536``
or ``131072`` for 64K or 128K. The length includes input and generated tokens;
reserve output space in the request. The script sets these memory controls:

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Option
     - Value
     - Purpose
   * - ``running_max_req_size``
     - 1
     - One active request; bounds request buffers.
   * - ``max_total_token_num``
     - context + 512
     - Fixed KV capacity, including internal padding.
   * - ``batch_max_tokens`` / ``chunked_prefill_size``
     - 512 / 512
     - Bounds prefill temporary tensors.
   * - ``linear_att_cache_size``
     - 2
     - Small hybrid prefix-state cache; recurrent states stay FP32.
   * - ``graph_max_batch_size``
     - 1
     - Capture decode batch one only.
   * - ``flashinfer_workspace_size``
     - 128 MiB
     - Reduce the workspace from its 512 MiB default for this single-request recipe.
   * - ``visual_infer_in_llm``
     - enabled
     - Serialize ViT and LLM in one GPU worker.
   * - ``max_image_token_count``
     - 256
     - Bounds image activation memory; image/video tokens count toward context.

Inline vision currently supports normal mode, TP=DP=1, one running request,
eager prefill and decode CUDA Graph. MTP, prefill CUDA Graph, microbatch overlap
and CPU KV offload are excluded from this recipe. Enable MTP only for a
checkpoint that actually contains matching MTP weights.

For 14 full-attention cache rows, four KV heads and head dimension 256, the
FP8 K/V payload at 64K / 128K / 256K is **1.75 / 3.50 / 7.00 GiB**, respectively.
The 512-token reserve adds about 0.014 GiB. These are KV payload sizes;
total GPU memory also includes weights, hybrid states, graph pools, CUDA
libraries, vision workspaces and temporary tensors.

Measured Results
~~~~~~~~~~~~~~~~

Each row below uses one RTX 4090D (24 GiB), one active request, the same Loop
AWQ export, static per-tensor FP8 KV, FP32 recurrent states, BF16 ViT in the LLM
worker, 128 MiB FlashInfer workspace and decode CUDA Graph. Context includes
input plus 128 generated tokens. Input repeats WorkBuddy task text with prefix
cache reuse disabled; a text request and a 512x512 blue-image request also pass
before each context test. ViT remains resident throughout.

.. list-table::
   :header-rows: 1
   :widths: 12 20 22 22 24

   * - Context
     - Input tokens
     - NVML peak (GiB)
     - Time to first token (s)
     - Decode (token/s)
   * - 64K
     - 65,408
     - 9.44
     - 24.54
     - 59.32
   * - 128K
     - 130,944
     - 11.19
     - 64.09
     - 53.48
   * - 256K
     - 262,016
     - 14.69
     - 196.32
     - 44.24

These are single-run capacity and timing measurements, with whole-card NVML
memory sampled every 10 ms during startup warmup, image inference, prefill and
decode. Decode rate counts the 127 tokens after the first token. The tested
environment uses PyTorch 2.8.0+cu128, Transformers 5.8.0, FlashInfer 0.5.3 and
vLLM commit ``a5f89ae29``. Checkpoint, software versions and image budgets affect
the measured peak and speed.

Enable MTP
----------

If the checkpoint includes MTP weights, add the following options to enable MTP:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0,1,2,3 LOADWORKER=4 LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 \
    python -m lightllm.server.api_server \
        --model_dir /path/to/SenseNova-Loop \
        --model_name sensenova_loop \
        --mtp_mode eagle_with_att \
        --mtp_step 3 \
        --tp 4 \
        --port 8000

2P1D Deployment
---------------

This script uses eight GPUs to start two TP2 prefill services and one TP4 decode service, with MTP enabled:

.. code-block:: bash

    MODEL_DIR=/path/to/SenseNova-Loop PORT=8000 \
    bash test/start_scripts/sensenova_loop/sensenova_loop_pd_2p1d.sh

Set ``GPU_IDS`` to select the GPUs, or add ``DRY_RUN=1`` to preview the commands.
Press ``Ctrl+C`` to stop all services.

Example Request
---------------

.. code-block:: bash

    curl http://localhost:8000/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{
          "model": "sensenova_loop",
          "messages": [{"role": "user", "content": "Hello, introduce yourself in one sentence."}],
          "max_tokens": 128
        }'
