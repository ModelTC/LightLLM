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

DFlash2 Speculative Decoding
---------------------------

DFlash2 generates candidate tokens with a draft model and verifies them with
SenseNova-Loop. Download the matching target and draft checkpoints:

.. code-block:: bash

    hf download sensenova/SenseNova-Loop-9B \
        --revision 3cc220e9c14bd7a74813c1299765e22147b1a3d5 \
        --local-dir /path/to/SenseNova-Loop-9B
    hf download sensenova/SenseNova-Loop-9B-DFlash2 \
        --local-dir /path/to/SenseNova-Loop-9B-DFlash2

The draft checkpoint is available at
`SenseNova-Loop-9B-DFlash2 <https://huggingface.co/sensenova/SenseNova-Loop-9B-DFlash2>`_.
The following example uses tensor parallelism across four GPUs,
BF16 model loading, FP32 linear-attention state, and five draft tokens per
verification round:

.. code-block:: bash

    MODEL_DIR=/path/to/SenseNova-Loop-9B \
    DRAFT_MODEL_DIR=/path/to/SenseNova-Loop-9B-DFlash2 \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    bash test/start_scripts/sensenova_loop/sensenova_loop_dflash2.sh

``MODEL_DIR`` and ``DRAFT_MODEL_DIR`` select compatible target and draft model
directories. Set ``DRY_RUN=1`` to print the launch command, ``METHOD=target`` to
disable drafting, or ``MTP_STEP=7`` to propose seven draft tokens per round.
``PYTHON`` selects the Python interpreter.

The selected target layers, inclusion of the final normalized hidden state,
and projection input width must match the draft checkpoint's training
configuration. See
:doc:`../references/draft_checkpoint_features` for ``target_feature_sources``.
The launch script enables per-request acceptance statistics in ``normal``
serving mode; see
:doc:`../references/speculative_acceptance_metrics` for the response schema.
