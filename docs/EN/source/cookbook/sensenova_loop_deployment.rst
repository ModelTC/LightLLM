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

Enable MTP
----------

The checkpoint includes MTP weights. Add the following options to enable MTP:

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
