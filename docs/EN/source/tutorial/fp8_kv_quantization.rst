.. _tutorial/fp8_kv_quantization_en:

FP8 KV Quantization and Calibration Guide
=========================================

This chapter describes FP8 KV inference in LightLLM, including:

- Running inference with calibration data (``fp8kv``)
- Quantization granularity differences between FA3 and FlashInfer
- Common errors and troubleshooting

Overview
--------

LightLLM FP8 KV inference requires a prepared calibration file (``kv_cache_calib.json``),
which is loaded by ``--kv_quant_calibration_config_path``.
You can use calibration files provided in ``test/advanced_config/``,
export one with `LightCompress <https://github.com/ModelTC/LightCompress>`_, or use your own compatible file.

Backend and Quantization Granularity
------------------------------------

Current behavior:

- ``fa3``: ``per_head`` scales (independent scale per head)
- ``flashinfer``: ``per_tensor`` scales (one scalar for K and one scalar for V)

Calibration files are backend-dependent:

- ``per_head`` files for ``fa3`` should be used with ``fa3`` inference.
- ``per_tensor`` files for ``flashinfer`` should be used with ``flashinfer`` inference.

Avoid mixing calibration files across different backends.

Start FP8 Inference with Calibration
------------------------------------

Inference mode example (FA3):

.. code-block:: console

    $ python -m lightllm.server.api_server \
        --model_dir /path/to/model \
        --llm_kv_type fp8kv \
        --llm_prefill_att_backend fa3 \
        --llm_decode_att_backend fa3 \
        --kv_quant_calibration_config_path /path/to/kv_cache_calib.json

Inference mode example (FlashInfer):

.. code-block:: console

    $ python -m lightllm.server.api_server \
        --model_dir /path/to/model \
        --llm_kv_type fp8kv \
        --llm_prefill_att_backend flashinfer \
        --llm_decode_att_backend flashinfer \
        --kv_quant_calibration_config_path /path/to/kv_cache_calib.json

Notes:

- ``fp8kv`` requires ``--kv_quant_calibration_config_path``.
- Keep the inference backend consistent with the backend expected by the calibration file.

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

   You are using ``--llm_kv_type fp8kv`` without a calibration file path.

2. ``quant_type not match`` error

   Usually caused by backend/file mismatch (for example, using a ``per_head`` file with ``flashinfer``).

3. Abnormal quality after backend switch

   Use a calibration file that matches the target backend instead of reusing an incompatible file.
