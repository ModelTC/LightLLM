.. _tutorial/fp8_kv_quantization_en:

FP8 KV Quantization and Calibration Guide
=========================================

This chapter describes FP8 KV inference in LightLLM, including:

- Running inference with calibration data (``fp8kv_sph`` or ``fp8kv_spt``)
- FP8 static per-head and per-tensor quantization modes
- Common errors and troubleshooting

Overview
--------

LightLLM FP8 KV inference requires a prepared calibration file (``kv_cache_calib.json``),
which is loaded by ``--kv_quant_calibration_config_path``.
You can use calibration files provided in ``test/advanced_config/``,
export one with `LightCompress <https://github.com/ModelTC/LightCompress>`_, or use your own compatible file.

Quantization Modes and Backend Mapping
------------------------------------------

LightLLM supports two FP8 KV quantization modes:

- ``fp8kv_sph``: FP8 Static Per-Head quantization, independent scale per head, uses ``fa3`` backend
- ``fp8kv_spt``: FP8 Static Per-Tensor quantization, one scalar for K and one scalar for V, uses ``flashinfer`` backend

Calibration files are mode-dependent:

- ``fp8kv_sph`` corresponds to ``per_head`` calibration files
- ``fp8kv_spt`` corresponds to ``per_tensor`` calibration files

Avoid mixing calibration files across different modes.

Start FP8 Inference with Calibration
------------------------------------

Inference mode example:

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

Notes:

- ``fp8kv_sph`` and ``fp8kv_spt`` require ``--kv_quant_calibration_config_path``.
- The attention backend will be automatically selected based on the quantization mode, no need to manually specify.

.. note::

   When using ``fp8kv_spt`` mode (FP8 static per-tensor quantization with flashinfer backend), 
   you must install ``flashinfer-python==0.6.5``. The default installed version is 0.6.3, 
   which may cause runtime issues. Install the correct version with:

   .. code-block:: console

       $ pip install flashinfer-python==0.6.5

Calibration File Schema
-----------------------

Key fields in ``kv_cache_calib.json``:

- ``quant_type``: ``per_head`` or ``per_tensor``
- ``num_layers``: number of layers
- ``num_head``: total number of heads
- ``scales_shape``: shape of the scale tensor
- ``scales``: actual scale values
- ``qmin`` / ``qmax``: FP8 numeric range parameters

At load time, LightLLM validates architecture, layer count, head count, and quantization type.

Multi-GPU Note
--------------

In multi-GPU (TP) setups, LightLLM slices the global scales to local rank heads automatically.
You only need to provide one full ``kv_cache_calib.json`` file.

Common Issues
-------------

1. Error says ``--kv_quant_calibration_config_path`` is required

   You are using ``--llm_kv_type fp8kv_sph`` or ``fp8kv_spt`` without a calibration file path.

2. ``quant_type not match`` error

   Usually caused by quantization mode/file mismatch (for example, using a ``per_tensor`` file with ``fp8kv_sph``).

3. Abnormal quality after mode switch

   Use a calibration file that matches the target quantization mode instead of reusing an incompatible file.

Different Target and Draft Head Layouts
--------------------------------------

For ``fp8kv_sph``, a shared KV buffer may contain layers with different head
layouts. Supply ``layouts`` instead of the top-level ``num_head``, ``scales``
and ``q_calibration`` fields. Each entry describes consecutive physical KV
layers: target full-attention layers first, followed by each draft's layers
in configured order. Linear-attention layers are not counted.

Keep ``num_layers``, ``num_target_layers`` and ``num_draft_layers`` at the top
level. For a target with 16 full-attention layers and a five-layer draft,
these are 21, 16 and 5. The two layout entries contain:

.. list-table:: Per-layout calibration fields
   :header-rows: 1

   * - Field
     - Target
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

Each entry's ``scales`` contains its independently calibrated K heads followed
by V heads. ``q_calibration.scales`` contains one Q scale per KV-head group,
not per query head. Values must be finite and positive; all declared layers
must be covered. TP slicing uses each entry's own head count.

Use the same file on P and D nodes. Target-only servers may also load this file
and use only the target rows. An old uniform-head calibration file cannot supply
independent eight-head draft scales for a four-head target; requesting that
layout now fails explicitly instead of duplicating scale values.
