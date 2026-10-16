.. _tutorial/fp8_kv_quantization_cn:

FP8 KV 量化与校准指南
======================

本章节介绍 LightLLM 中 FP8 KV 推理的使用方式，包括：

- 使用校准文件进行推理（``fp8kv_sph`` 或 ``fp8kv_spt``）
- FP8 静态按 head 和按 tensor 的量化模式
- 常见报错与排查建议

功能概览
--------

LightLLM 的 FP8 KV 推理需要准备好的校准文件（``kv_cache_calib.json``），
并通过 ``--kv_quant_calibration_config_path`` 加载。
你可以直接使用 ``test/advanced_config/`` 目录下已有的校准文件，
也可以使用 `LightCompress <https://github.com/ModelTC/LightCompress>`_ 工具导出，或使用自有兼容文件。

量化模式与后端对应
------------------

LightLLM 支持两种 FP8 KV 量化模式：

- ``fp8kv_sph``: FP8 静态按 head 量化（Static Per-Head），每个 head 独立 scale，对应 ``fa3`` 后端
- ``fp8kv_spt``: FP8 静态按 tensor 量化（Static Per-Tensor），K/V 各一个标量 scale，对应 ``flashinfer`` 后端

校准文件与量化模式强相关：

- ``fp8kv_sph`` 对应 ``per_head`` 校准文件
- ``fp8kv_spt`` 对应 ``per_tensor`` 校准文件

不建议混用不同模式的校准文件。

使用校准文件启动 FP8 推理
-------------------------

推理模式示例：

.. code-block:: console

    $ python -m lightllm.server.api_server \
        --model_dir /path/to/model \
        --llm_kv_type fp8kv_sph \
        --kv_quant_calibration_config_path /path/to/kv_cache_calib.json

.. code-block:: console

    $ python -m lightllm.server.api_server \
        --model_dir /path/to/model \
        --llm_kv_type fp8kv_spt \
        --kv_quant_calibration_config_path /path/to/kv_cache_calib.json

说明：

- ``fp8kv_sph`` 和 ``fp8kv_spt`` 模式必须提供 ``--kv_quant_calibration_config_path``。
- attention backend 会根据量化模式自动选择，无需手动指定。

.. note::

   使用 ``fp8kv_spt`` 模式（FP8 静态按 tensor 量化，使用 flashinfer 后端）时，
   必须安装 ``flashinfer-python==0.6.5``。默认安装的版本是 0.6.3，
   可能导致运行错误。请使用以下命令安装正确版本：

   .. code-block:: console

       $ pip install flashinfer-python==0.6.5

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

   说明你使用了 ``--llm_kv_type fp8kv_sph`` 或 ``fp8kv_spt`` 但未传入校准文件路径。

2. 报错 ``quant_type not match``

   通常是量化模式与校准文件类型不一致。例如拿 ``per_tensor`` 文件去跑 ``fp8kv_sph``。

3. 切换量化模式后效果异常

   建议使用与目标量化模式匹配的校准文件，不要跨模式复用不兼容文件。

主模型与草稿模型的 head 布局不同
--------------------------------

使用 ``fp8kv_sph`` 时，共享 KV buffer 中的层可以使用不同的 head 布局。
用 ``layouts`` 列表代替顶层的 ``num_head``、``scales`` 和 ``q_calibration``。
每个条目描述连续的物理 KV 层：先排列主模型 full-attention 层，再按配置顺序
排列各个 draft 的层，不计入 linear-attention 层。

顶层保留 ``num_layers``、``num_target_layers`` 和 ``num_draft_layers``。
例如主模型有 16 个 full-attention 层、draft 有 5 层时，这三个值分别为 21、16、5。
两个 layout 条目分别包含：

.. list-table:: 各布局的校准字段
   :header-rows: 1

   * - 字段
     - 主模型
     - Draft
   * - ``num_layers``
     - 16
     - 5
   * - ``num_head`` / ``head_dim``
     - 4 / 256
     - 8 / 128
   * - ``scales_shape``
     - [16, 8]
     - [5, 16]
   * - ``q_calibration.num_head``
     - 4
     - 8
   * - ``q_calibration.scales_shape``
     - [16, 4]
     - [5, 8]

各条目的 ``scales`` 保存对应模型独立校准的 K head scale，然后是 V head scale。
``q_calibration.scales`` 为每个 KV-head group 保存一个 Q scale，并非每个 query head 一个。
所有值必须有限且为正数，条目必须覆盖声明的全部层；TP 切分使用各条目自己的 head 数。

P、D 节点使用同一份文件。不开启 draft 的服务也可以加载该文件，仅使用主模型行。
旧的统一 head 校准文件无法为 4-head 主模型提供独立的 8-head draft scale；
请求这种布局会明确报错，不再自动重复 scale。
