.. _sensenova_loop_deployment:

SenseNova Loop 部署指南
===========================

``--loop_cache_mode`` 控制各轮 Loop 的 KV、conv 和 SSM 状态：

* ``independent``（默认）：每个执行层使用独立缓存。
* ``overwrite``：同一物理层的各轮执行复用并覆盖缓存。启用方式为
  ``--loop_cache_mode overwrite``，推理结果可能与独立缓存模式不同。

两种模式的前向执行次数均由 ``loop_settings`` 决定。例如 32 个物理层、循环区间
``[4, 28)``、执行 3 轮时，共执行 80 次。独立模式分配 20 层 full attention
和 60 层 linear attention 缓存，覆盖模式分配 8 层和 24 层。

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

AWQ 部署
--------

在仓库根目录执行：

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 LOADWORKER=1 \
    python -m lightllm.server.api_server \
        --model_dir /path/to/SenseNova-Loop-9B-AWQ \
        --model_name sensenova_loop \
        --tp 1 \
        --disable_vision \
        --disable_audio \
        --port 8000

多卡部署时，通过 ``CUDA_VISIBLE_DEVICES`` 选择显卡，并将 ``--tp`` 设为对应卡数。

单卡长上下文可使用 FP8 KV 启动脚本：

.. code-block:: bash

    MODEL_DIR=/path/to/SenseNova-Loop-9B-AWQ \
    KV_CALIB=test/advanced_config/fp8_calibration_per_tensor/kv_cache_calib_per_tensor_sensenova_loop.json \
    CUDA_VISIBLE_DEVICES=0 CONTEXT_LENGTH=262144 PORT=8000 \
    bash test/start_scripts/sensenova_loop/sensenova_loop_4090.sh

``CONTEXT_LENGTH`` 可设为 ``65536``、``131072`` 或 ``262144``，包含输入和输出 token。
``PORT`` 指定服务端口，添加 ``DRY_RUN=1`` 可查看完整启动命令。
脚本启用 FP8 KV 和解码 CUDA Graph，使用单卡、单个运行请求，禁用视觉、音频和 MTP。

示例标定文件使用 SenseNova Loop BF16 模型和 WorkBuddy 数据生成，适用于
``independent`` 模式、Loop 两轮。``overwrite`` 模式需按物理层缓存布局重新标定，
并通过 ``KV_CALIB`` 指定对应文件。
FP8 KV 参数说明见 :doc:`../tutorial/fp8_kv_quantization`。

启用 MTP
--------

如果 checkpoint 包含 MTP 权重，增加以下参数即可启用：

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
设置 ``LOOP_CACHE_MODE=overwrite`` 可开启 KV、conv 和 SSM 状态覆盖模式，
默认为 ``independent``。该设置会统一传给两个 Prefill、Decode 和 PD Master。
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
