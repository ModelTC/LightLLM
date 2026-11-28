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

示例中的校准文件使用 SenseNova Loop BF16 模型和 WorkBuddy 数据生成，适用于 Loop 两轮。
其他模型请替换 ``KV_CALIB``；FP8 KV 参数说明见 :doc:`../tutorial/fp8_kv_quantization`。

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

DFlash2 推测解码
----------------

DFlash2 使用草稿模型并行生成候选 token，由 SenseNova-Loop 主模型验证。
先下载匹配的主模型与草稿权重：

.. code-block:: bash

    hf download sensenova/SenseNova-Loop-9B \
        --revision 3cc220e9c14bd7a74813c1299765e22147b1a3d5 \
        --local-dir /path/to/SenseNova-Loop-9B
    hf download sensenova/SenseNova-Loop-9B-DFlash2 \
        --local-dir /path/to/SenseNova-Loop-9B-DFlash2

草稿权重见
`SenseNova-Loop-9B-DFlash2 <https://huggingface.co/sensenova/SenseNova-Loop-9B-DFlash2>`_。
以下示例使用四张 GPU 进行张量并行，以 BF16 加载模型、以 FP32 保存线性注意力状态，
每轮提出 5 个草稿 token：

.. code-block:: bash

    MODEL_DIR=/path/to/SenseNova-Loop-9B \
    DRAFT_MODEL_DIR=/path/to/SenseNova-Loop-9B-DFlash2 \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    bash test/start_scripts/sensenova_loop/sensenova_loop_dflash2.sh

``MODEL_DIR`` 和 ``DRAFT_MODEL_DIR`` 分别指定兼容的主模型与草稿模型目录。
设置 ``DRY_RUN=1`` 可打印启动命令，``METHOD=target`` 可关闭草稿模型，
``MTP_STEP=7`` 可设置为每轮提出 7 个草稿 token。``PYTHON`` 用于指定 Python 解释器。

草稿配置中的特征层顺序、最终归一化特征和投影宽度须与训练配置一致。
包含最终归一化特征时，使用 ``target_feature_sources`` 显式声明。
该启动脚本在 ``normal`` 服务模式下返回每个请求的接受统计。
