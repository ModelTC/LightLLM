.. _tutorial/fp8_kv_quantization_cn:

FP8 KV 量化使用指南
===================

通过 ``--llm_kv_type`` 选择 FP8 KV Cache 量化模式。
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
