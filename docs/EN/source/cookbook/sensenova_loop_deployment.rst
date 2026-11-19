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

AWQ Deployment
--------------

Run from the repository root:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 LOADWORKER=1 \
    python -m lightllm.server.api_server \
        --model_dir /path/to/SenseNova-Loop-9B-AWQ \
        --model_name sensenova_loop \
        --tp 1 \
        --disable_vision \
        --disable_audio \
        --port 8000

For multiple GPUs, select them with ``CUDA_VISIBLE_DEVICES`` and set ``--tp``
to the corresponding GPU count.

For long contexts on one GPU, use the FP8 KV launch script:

.. code-block:: bash

    MODEL_DIR=/path/to/SenseNova-Loop-9B-AWQ \
    KV_CALIB=test/advanced_config/fp8_calibration_per_tensor/kv_cache_calib_per_tensor_sensenova_loop.json \
    CUDA_VISIBLE_DEVICES=0 CONTEXT_LENGTH=262144 PORT=8000 \
    bash test/start_scripts/sensenova_loop/sensenova_loop_4090.sh

Set ``CONTEXT_LENGTH`` to ``65536``, ``131072`` or ``262144``, including both
input and generated tokens. Use ``PORT`` to set the server port and add
``DRY_RUN=1`` to print the full command. The script enables FP8 KV and decode
CUDA Graph for one GPU and one active request, with vision, audio and MTP disabled.

The example's calibration file was generated with the SenseNova Loop BF16 model
and WorkBuddy data for two Loop passes. Use a matching ``KV_CALIB`` for other
models. See :doc:`../tutorial/fp8_kv_quantization` for FP8 KV options.

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
