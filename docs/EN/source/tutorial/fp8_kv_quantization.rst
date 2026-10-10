.. _tutorial/fp8_kv_quantization_en:

FP8 KV Quantization and Calibration Guide
=========================================

This chapter describes FP8 KV inference in LightLLM, including:

- Running inference with calibration data (``fp8kv_sph`` or ``fp8kv_spt``)
- FP8 static per-head and per-tensor quantization modes
- Common errors and troubleshooting

Overview
--------

LightLLM static FP8 KV inference requires a prepared calibration file (``kv_cache_calib.json``),
which is loaded by ``--kv_quant_calibration_config_path``.
You can use calibration files provided in ``test/advanced_config/``,
export one with `LightCompress <https://github.com/ModelTC/LightCompress>`_, or use your own compatible file.

Quantization Modes and Backend Mapping
------------------------------------------

LightLLM supports three FP8 KV quantization modes:

- ``fp8kv_sph``: FP8 Static Per-Head quantization, independent scale per head, uses ``fa3`` backend
- ``fp8kv_spt``: FP8 Static Per-Tensor quantization, one scalar for K and one scalar for V, uses ``flashinfer`` backend
- ``fp8kv_dsa``: dynamic per-token, per-128-element quantization for DeepSeek-V3.2 and GLM-5.3 Flash sparse MLA, without calibration

Calibration files are mode-dependent:

- ``fp8kv_sph`` corresponds to ``per_head`` calibration files
- ``fp8kv_spt`` corresponds to ``per_tensor`` calibration files

Avoid mixing calibration files across different modes.

GLM-5.3 Flash Dynamic FP8 KV
----------------------------

Both ``glm5_next`` and ``glm5_next_text`` support ``fp8kv_dsa`` with BF16 computation:

.. code-block:: console

    $ python -m lightllm.server.api_server \
        --model_dir /path/to/GLM-5.3-Flash --tp 4 \
        --data_type bfloat16 --llm_kv_type fp8kv_dsa

Sparse MLA KV uses E4M3 values with four FP32 scales. To reuse the FlashMLA
V3.2 sparse decode kernel, the cache and query include a zero-filled RoPE tail;
attention retains NoPE semantics. Including the indexer and alignment, each
sparse layer uses 800 bytes per token instead of 1168 bytes (31.5% less).
KDA convolution/SSM state and the indexer's raw request tails retain their
original computation dtypes, so this is not a 31.5% reduction in total model memory.

Prefill uses newly computed BF16 KV directly and dequantizes cached prefixes.
The same layout is used by native MTP draft layers, CPU prefix caching and P/D
transfer. P and D must both use ``--llm_kv_type fp8kv_dsa``. For the supplied
``test/start_scripts/glm53/glm53_pd_1p1d.sh`` launcher, set
``LLM_KV_TYPE=fp8kv_dsa``. Transfer pages must still fit the full KDA/indexer state;
increasing ``--pd_kv_page_size`` may be necessary because FP8 token pages are smaller.

This path requires the ``flash_mla`` package with V3.2 sparse FP8 decode support,
in addition to the existing GLM-5.3 Flash dependencies. ``fp8kv_sph`` and
``fp8kv_spt`` are not supported for this model.

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
