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

RTX 4090 / 4090D 单卡（24 GiB）
--------------------------------

单请求长上下文使用 Loop AWQ 导出：embedding、lm_head 与主要线性层为 INT4，
视觉塔保留 BF16，KV 使用 FP8，解码保留 CUDA Graph。ViT 在 LLM 进程内串行执行，
共用 CUDA 上下文与显存池；推理、调度与分页 KV 均走 LightLLM 原生路径。

本配方的导出由 ``config.json`` 中的 ``loop_awq_config`` 标识：
``bits=4``、``group_size=128``、``zero_point=true``、
``linear_layout="awq_gemm"``、``input_scale_mode="runtime"``、
``embedding_layout="row_major_int4"``，权重位于 ``unquantized.safetensors``
与 ``modules/*.safetensors``。模型目录还需保留 tokenizer、chat template 和图像处理器文件。
普通折叠缩放的 AWQ 导出仍走原有加载流程，与此导出格式不同。

环境需安装兼容的 vLLM（AWQ-Marlin）及支持 Ada 的 FlashInfer。
另外准备与此 checkpoint、循环次数匹配的 **per_tensor** FP8 KV 标定文件，
参见 :doc:`../tutorial/fp8_kv_quantization`。标定行按全注意力执行顺序排列：
32 个物理层中 [4, 28) 执行两遍时，共有 14 行独立的 KV 缓存。
其他循环次数或其他模型的标定文件不能直接套用。

在仓库根目录执行：

.. code-block:: bash

    MODEL_DIR=/path/to/SenseNova-Loop-AWQ \
    KV_CALIB=/path/to/kv_cache_calib_per_tensor.json \
    CUDA_VISIBLE_DEVICES=0 CONTEXT_LENGTH=262144 PORT=8000 \
    bash test/start_scripts/sensenova_loop/sensenova_loop_4090.sh

添加 ``DRY_RUN=1`` 可查看完整命令。64K、128K 分别设置
``CONTEXT_LENGTH=65536``、``131072``。上下文上限包含输入和输出 token，
请求需为输出预留空间。脚本显式指定以下显存相关参数：

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - 参数
     - 值
     - 作用
   * - ``running_max_req_size``
     - 1
     - 单个运行请求，约束请求缓冲区。
   * - ``max_total_token_num``
     - 上下文 + 512
     - 固定 KV 容量，包含内部占位与对齐预留。
   * - ``batch_max_tokens`` / ``chunked_prefill_size``
     - 512 / 512
     - 控制预填充临时张量大小。
   * - ``linear_att_cache_size``
     - 2
     - 限制混合注意力前缀状态缓存，递归状态保留 FP32。
   * - ``graph_max_batch_size``
     - 1
     - 仅捕获 batch=1 的解码图。
   * - ``flashinfer_workspace_size``
     - 128 MiB
     - 单请求使用较小工作区，默认值为 512 MiB。
   * - ``visual_infer_in_llm``
     - 开启
     - ViT 与 LLM 在同一 GPU 进程内串行执行。
   * - ``max_image_token_count``
     - 256
     - 约束图像激活显存，图像与视频 token 也计入上下文。

ViT 同进程模式目前限定普通服务、TP=DP=1、单请求、eager 预填充，支持解码 CUDA Graph。
此配方不启用 MTP、预填充 CUDA Graph、微批重叠与 CPU KV 卸载。
仅在 checkpoint 确实包含匹配的 MTP 权重时启用 MTP。

按 14 个全注意力缓存槽、4 个 KV 头、head dimension=256 计算，
64K / 128K / 256K 的 FP8 K/V 数据分别为 **1.75 / 3.50 / 7.00 GiB**，
额外 512 token 预留约增加 0.014 GiB。这是 KV 数据量；整卡还包括权重、
混合注意力状态、CUDA Graph、CUDA 库、视觉工作区和临时张量。

实测结果
~~~~~~~~

每档使用单张 RTX 4090D（24 GiB）、单个运行请求、同一份 Loop AWQ 导出、
静态 per-tensor FP8 KV、FP32 递归状态、LLM 同进程的 BF16 ViT、
128 MiB FlashInfer 工作区及解码 CUDA Graph。上下文包含输入与 128 个输出 token。
输入由 WorkBuddy 任务文本重复组成，关闭前缀缓存复用；每档还先验证文本请求与
512x512 蓝色图片请求，视觉塔全程保留。

.. list-table::
   :header-rows: 1
   :widths: 12 20 22 22 24

   * - 上下文
     - 输入 token
     - NVML 峰值（GiB）
     - 首 token 耗时（秒）
     - 解码（token/s）
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

以上为每档单次容量与速度测试，整卡显存通过 NVML 每 10 ms 采样，覆盖启动预热、
图像推理、预填充与解码；解码速度按首 token 之后的 127 个 token 计算。
测试环境为 PyTorch 2.8.0+cu128、Transformers 5.8.0、FlashInfer 0.5.3、
vLLM commit ``a5f89ae29``。具体 checkpoint、软件版本与图像预算会影响峰值和速度。

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
