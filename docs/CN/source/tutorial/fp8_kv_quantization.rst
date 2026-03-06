.. _tutorial/fp8_kv_quantization_cn:

FP8 KV 量化与校准指南
======================

本章节介绍 LightLLM 中 FP8 KV 量化的完整流程，包括：

- 导出校准文件（``--export_fp8kv_calibration``）
- 使用校准文件进行推理（``fp8kv``）
- FA3 与 FlashInfer 后端下的量化粒度差异
- 常见报错与排查建议

功能概览
--------

LightLLM 的 FP8 KV 量化采用离线校准方案：

1. 先运行导出模式，统计 KV 的最大绝对值并导出 ``kv_cache_calib.json``。
2. 再在推理模式加载该文件，将 KV 按 scale 量化为 ``float8_e4m3fn`` 存储。

后端与量化粒度
--------------

当前行为如下：

- ``fa3``: 使用 ``per_head``（每个 head 独立 scale）
- ``flashinfer``: 使用 ``per_tensor``（K/V 各一个标量 scale）

因此，校准文件与后端强相关：

- ``fa3`` 生成的 ``per_head`` 校准文件用于 ``fa3`` 推理。
- ``flashinfer`` 生成的 ``per_tensor`` 校准文件用于 ``flashinfer`` 推理。

不建议混用不同后端导出的校准文件。

步骤一：导出校准文件
--------------------

导出模式示例（FA3）：

.. code-block:: console

    $ python -m lightllm.server.api_server \
        --model_dir /path/to/model \
        --export_fp8kv_calibration \
        --llm_prefill_att_backend fa3 \
        --llm_decode_att_backend fa3 \
        --disable_cudagraph

导出模式示例（FlashInfer）：

.. code-block:: console

    $ python -m lightllm.server.api_server \
        --model_dir /path/to/model \
        --export_fp8kv_calibration \
        --llm_prefill_att_backend flashinfer \
        --llm_decode_att_backend flashinfer \
        --disable_cudagraph

说明：

- 设置 ``--export_fp8kv_calibration`` 后，会在运行过程中收集 KV 统计信息。
- 校准完成后，会在当前工作目录输出 ``kv_cache_calib.json``。
- 导出模式要求 ``--disable_cudagraph``，且 ``--llm_kv_type`` 保持为 ``None``。
- 仓库 ``test/advanced_config/`` 目录中已存放常用模型的校准文件，可按需直接使用或作为参考。

使用 benchmark_qps.py 进行随机数据校准
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

除了在线业务流量，也可以使用 ``test/benchmark/service/benchmark_qps.py`` 工具构造随机请求进行校准。

- 默认累计约 4000 次推理后会输出一次校准结果。
- 实践中可执行以下命令两次，以更稳定地覆盖统计范围。

示例命令：

.. code-block:: console

    $ python test/benchmark/service/benchmark_qps.py --url http://127.0.0.1:8000/generate_stream --tokenizer_path ../Qwen3-30B-A3B --input_len 1000 --output_len 2000 --input_qps 10 --input_num 200 --range_ratio 0.9

步骤二：使用校准文件启动 FP8 推理
---------------------------------

推理模式示例（FA3）：

.. code-block:: console

    $ python -m lightllm.server.api_server \
        --model_dir /path/to/model \
        --llm_kv_type fp8kv \
        --llm_prefill_att_backend fa3 \
        --llm_decode_att_backend fa3 \
        --kv_quant_calibration_config_path /path/to/kv_cache_calib.json

推理模式示例（FlashInfer）：

.. code-block:: console

    $ python -m lightllm.server.api_server \
        --model_dir /path/to/model \
        --llm_kv_type fp8kv \
        --llm_prefill_att_backend flashinfer \
        --llm_decode_att_backend flashinfer \
        --kv_quant_calibration_config_path /path/to/kv_cache_calib.json

说明：

- ``fp8kv`` 模式必须提供 ``--kv_quant_calibration_config_path``。
- 建议推理时的 attention backend 与导出校准时保持一致。

校准文件格式
------------

导出的 ``kv_cache_calib.json`` 主要字段包括：

- ``quant_type``: ``per_head`` 或 ``per_tensor``
- ``num_layers``: 层数
- ``num_head``: 总 head 数
- ``scales_shape``: scale 张量形状
- ``scales``: 实际 scale 数值
- ``qmin`` / ``qmax``: FP8 范围参数

加载校准文件时，会校验模型架构、层数、head 数及量化类型是否匹配。

多卡说明
--------

在多卡（TP）场景下，系统会根据当前 rank 自动切分本地需要的 head 对应 scale。
你仍然只需要提供一份全量 ``kv_cache_calib.json``。

常见问题
--------

1. 启动时报错需要 ``--kv_quant_calibration_config_path``

   说明你使用了 ``--llm_kv_type fp8kv`` 但未传入校准文件路径。

2. 启动时报错要求 ``--disable_cudagraph``

    说明你使用了 ``--export_fp8kv_calibration``，该模式必须禁用 cudagraph。

3. 报错 ``quant_type not match``

   通常是后端与校准文件类型不一致。例如拿 ``per_head`` 文件去跑 ``flashinfer``。

4. 切换后端后效果异常

   建议按目标后端重新导出校准文件，不要跨后端复用。
