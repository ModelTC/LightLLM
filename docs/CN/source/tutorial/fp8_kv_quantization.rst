.. _tutorial/fp8_kv_quantization_cn:

KV Cache 量化使用指南
=====================

通过 ``--llm_kv_type`` 选择 KV Cache 量化模式。
运行前请按 :doc:`安装指南 <../getting_started/installation>` 准备环境。

选择量化模式
------------

.. list-table::
   :header-rows: 1

   * - 模式
     - 适用场景
     - 校准文件
   * - ``fp8kv_sph``
     - 静态按 head 量化，需 Hopper GPU
     - ``per_head``，须包含 Q 校准数据
   * - ``fp8kv_spt``
     - 静态按 tensor 量化，需 FlashInfer
     - ``per_tensor``
   * - ``fp8kv_dsa``
     - DeepSeek-V3.2、GLM-5.3 Flash
     - 不需要
   * - ``nvfp4kv``
     - 动态 NVFP4；MLA 使用方法见下文
     - 不需要

静态 FP8 KV
-----------

准备与模型和量化模式匹配的 ``kv_cache_calib.json``，然后指定文件路径启动。

按 head 量化：

.. code-block:: console

    $ python -m lightllm.server.api_server \
        --model_dir /path/to/model \
        --llm_kv_type fp8kv_sph \
        --kv_quant_calibration_config_path /path/to/per_head/kv_cache_calib.json

按 tensor 量化：

.. code-block:: console

    $ python -m lightllm.server.api_server \
        --model_dir /path/to/model \
        --llm_kv_type fp8kv_spt \
        --kv_quant_calibration_config_path /path/to/per_tensor/kv_cache_calib.json

两种模式都会自动选择注意力后端，无需额外指定。
多卡部署时添加 ``--tp <卡数>``，仍只需提供一份完整的校准文件。

动态 FP8 KV
-----------

DeepSeek-V3.2 和 GLM-5.3 Flash 使用 ``--llm_kv_type fp8kv_dsa``，无需提供校准文件。
环境中需安装兼容的 FlashMLA；可按安装指南从本仓库的 Dockerfile 构建镜像。
源码安装用户还需单独安装 FlashMLA，仅安装 ``requirements.txt`` 不够。

以下以 GLM-5.3 Flash 为例，该模型须使用 ``bfloat16``，FP8 KV 模式仅支持 ``fp8kv_dsa``：

.. code-block:: console

    $ python -m lightllm.server.api_server \
        --model_dir /path/to/GLM-5.3-Flash --tp 4 \
        --data_type bfloat16 --llm_kv_type fp8kv_dsa

NVFP4 MLA KV
------------

普通 DeepSeek/GLM MLA 和 GLM-5.3 Flash 的稀疏 MLA 可以使用 ``--llm_kv_type nvfp4kv``。
latent 按 16 个元素动态量化为 E2M1，每组保存一个 E4M3 scale，无需离线校准。
Q 保持计算精度；带 RoPE 的 MLA 保留 RoPE 原始精度。普通 MLA 使用 ``triton`` 后端，
在计算 tile 内解码历史 KV。

GLM-5.3 Flash 使用 BF16 计算，并保留现有 K-pool 索引格式和线性注意力状态：

.. code-block:: console

    $ python -m lightllm.server.api_server \
        --model_dir /path/to/GLM-5.3-Flash --disable_vision --tp 4 \
        --data_type bfloat16 --llm_kv_type nvfp4kv --page_size 16 \
        --chunked_prefill_size 4096

GLM 的 NVFP4 注意力路径需要 Hopper GPU 及兼容的 FlashMLA 和 FA3，安装方式见安装指南。
KV 按压缩格式保存，计算时按需使用临时 BF16 workspace。
实际可加载 token 容量取决于权重、运行态和显存预算。
GLM 的 KV 页须为 4 的倍数；默认 ``--page_size 1`` 启动时自动调整为 4，
也可以显式指定 16 等更大的页。CPU/P-D 页同样须按 4 对齐。
GLM 的原生 MTP 与主模型使用相同的 NVFP4 编码。

DeepSeek-V4 NVFP4 KV
-------------------

DeepSeek-V4 可以使用 ``--llm_kv_type nvfp4kv``，需要 BF16 计算、Hopper GPU
及兼容的 FA3。依赖安装方式见安装指南；仓库 Dockerfile 固定了兼容的 FlashMLA 版本。
以下示例使用 FP8 权重；权重量化与 KV 量化分别配置：

.. code-block:: bash

    $ python -m lightllm.server.api_server \
        --model_dir /path/to/DeepSeek-V4-Flash-FP8 --tp 8 --dp 8 \
        --enable_ep_moe --expert_dtype fp8 --data_type bfloat16 \
        --llm_kv_type nvfp4kv --chunked_prefill_size 4096

滑窗和 C4/C128 压缩历史的非 RoPE 特征按 block16 E2M1/E4M3 编码，RoPE 保留 BF16。
indexer 的 FP8 key/FP32 scale 及压缩器续算状态保持原有格式，并保留 attention sink 语义。
注意力只在固定容量的临时区解码所选 KV，不保留完整历史的 BF16 副本。
原生 MTP 和 DSpark 的 KV 状态使用同一编码入口。

默认 ``--page_size 1`` 自动调整为 256；显式设置的 KV 页、CPU 缓存页和 P/D 传输页
须为 256 的倍数。P/D 两端须使用相同的 KV 编码。
V4 的总容量收益取决于层的压缩比例、BF16 RoPE、indexer 和滑窗，不能直接套用普通
GQA 的容量比例。其他模型的精度测试结果也不能替代 V4 的任务精度评估。

P/D 分离部署
------------

使用动态 FP8 KV 进行 P/D 分离部署时，Prefill 和 Decode 两端都须设置 ``--llm_kv_type fp8kv_dsa``。
GLM-5.3 Flash 可使用单机 8 卡脚本，P/D 各使用 4 卡：

.. code-block:: console

    $ LLM_KV_TYPE=fp8kv_dsa bash test/start_scripts/glm53/glm53_pd_1p1d.sh \
        8088 <node_ip> /path/to/GLM-5.3-Flash

将 ``<node_ip>`` 替换为服务可访问的本机 IP。脚本会同时配置两端的量化模式和传输页大小。
手动启动时，两端须使用相同的量化模式和 ``--pd_kv_page_size``。

常见问题
--------

- **提示缺少校准文件或校准类型不匹配**：静态模式必须传入
  ``--kv_quant_calibration_config_path``，文件须匹配模型及所选模式；
  ``fp8kv_sph`` 文件还须包含 Q 校准数据。
- **提示缺少依赖**：按当前安装指南安装依赖；动态模式还需兼容的 FlashMLA。
- **P/D 传输页容量不足**：同时增大两端的 ``--pd_kv_page_size``，并确保其为
  ``--page_size`` 的整数倍。
