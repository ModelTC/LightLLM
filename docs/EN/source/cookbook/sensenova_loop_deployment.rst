.. _sensenova_loop_deployment:

SenseNova Loop Deployment Guide
==============================

``--loop_cache_mode`` controls KV, conv, and SSM states across Loop passes:

* ``independent`` (default): each execution layer uses separate caches.
* ``overwrite``: visits to the same physical layer reuse and overwrite its caches.
  Enable it with ``--loop_cache_mode overwrite``. Inference results may
  differ from independent mode.

In overwrite mode, each linear attention visit starts from the conv/SSM state
saved before the current forward. Only the final visit writes back to persistent
caches. Intermediate visits use temporary state for the current batch, so later
visits do not consume state produced by earlier visits in the same forward.
This mode requires ``--ssm_state_mode native`` (the default).

``loop_settings`` determines the number of forward executions in both modes.
For example, 32 physical layers with three passes over ``[4, 28)`` execute 80 layer
visits. Independent mode allocates 20 full attention and 60 linear attention cache
layers; overwrite mode allocates 8 and 24, respectively.

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

The example calibration file was generated with the SenseNova Loop BF16 model and
WorkBuddy data for two Loop passes in ``independent`` mode. For ``overwrite`` mode,
generate a calibration file for the physical-layer cache layout and select it with
``KV_CALIB``. See :doc:`../tutorial/fp8_kv_quantization` for FP8 KV options.

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
Set ``MTP_DRAFT_MODEL_DIR`` to use a separate draft checkpoint; it defaults to
``MODEL_DIR``. Both prefill services and the decode service load that draft in
``eagle_with_att`` mode.
Set ``LOOP_CACHE_MODE=overwrite`` to reuse and overwrite KV, conv, and SSM states;
the default is ``independent``. The setting applies to both prefill services, the
decode service, and the PD master.
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
