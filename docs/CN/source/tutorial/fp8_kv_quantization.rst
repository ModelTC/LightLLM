.. _tutorial/fp8_kv_quantization_cn:

FP8 KV 量化与校准指南
======================

本章节介绍 LightLLM 中 FP8 KV 推理的使用方式，包括：

- 使用校准文件进行推理（``fp8kv``）
- FA3 与 FlashInfer 后端下的量化粒度差异
- 常见报错与排查建议

功能概览
--------

LightLLM 的 FP8 KV 推理需要准备好的校准文件（``kv_cache_calib.json``），
并通过 ``--kv_quant_calibration_config_path`` 加载。
你可以直接使用 ``test/advanced_config/`` 目录下已有的校准文件，
也可以使用 `LightCompress <https://github.com/ModelTC/LightCompress>`_ 工具导出，或使用自有兼容文件。

后端与量化粒度
--------------

当前行为如下：

- ``fa3``: 使用 ``per_head``（每个 head 独立 scale）
- ``flashinfer``: 使用 ``per_tensor``（K/V 各一个标量 scale）

因此，校准文件与后端强相关：

- ``fa3`` 对应 ``per_head`` 校准文件，应配合 ``fa3`` 推理。
- ``flashinfer`` 对应 ``per_tensor`` 校准文件，应配合 ``flashinfer`` 推理。

不建议混用不同后端的校准文件。

使用校准文件启动 FP8 推理
-------------------------

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
- 建议推理时的 attention backend 与校准文件要求保持一致。

校准文件格式
------------

``kv_cache_calib.json`` 主要字段包括：

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

2. 报错 ``quant_type not match``

   通常是后端与校准文件类型不一致。例如拿 ``per_head`` 文件去跑 ``flashinfer``。

3. 切换后端后效果异常

   建议使用与目标后端匹配的校准文件，不要跨后端复用不兼容文件。
