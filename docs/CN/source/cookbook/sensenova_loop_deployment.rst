.. _sensenova_loop_deployment:

SenseNova Loop 部署指南
===========================

启动服务
--------

下面以 4 卡部署为例，在仓库根目录执行：

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0,1,2,3 LOADWORKER=4 LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 \
    python -m lightllm.server.api_server \
        --model_dir /path/to/SenseNova-Loop \
        --model_name sensenova_loop \
        --tp 4 \
        --port 8000

启用 MTP
--------

checkpoint 已包含 MTP 权重，增加以下参数即可启用：

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0,1,2,3 LOADWORKER=4 LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 \
    python -m lightllm.server.api_server \
        --model_dir /path/to/SenseNova-Loop \
        --model_name sensenova_loop \
        --mtp_mode eagle_with_att \
        --mtp_step 3 \
        --tp 4 \
        --port 8000

2P1D 部署
---------

下面的脚本默认使用 8 张 GPU，启动两个 TP2 Prefill 服务和一个 TP4 Decode 服务，并启用 MTP：

.. code-block:: bash

    MODEL_DIR=/path/to/SenseNova-Loop PORT=8000 \
    bash test/start_scripts/sensenova_loop/sensenova_loop_pd_2p1d.sh

可通过 ``GPU_IDS`` 指定显卡列表，添加 ``DRY_RUN=1`` 可先查看启动命令。
按 ``Ctrl+C`` 会关闭所有服务。

请求示例
--------

.. code-block:: bash

    curl http://localhost:8000/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{
          "model": "sensenova_loop",
          "messages": [{"role": "user", "content": "你好，请用一句话介绍自己。"}],
          "max_tokens": 128
        }'
