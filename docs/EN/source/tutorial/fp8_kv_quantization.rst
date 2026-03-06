.. _tutorial/fp8_kv_quantization_en:

FP8 KV Quantization and Calibration Guide
=========================================

This chapter describes the end-to-end FP8 KV quantization workflow in LightLLM, including:

- Exporting calibration data (``--export_fp8kv_calibration``)
- Running inference with calibration data (``fp8kv``)
- Quantization granularity differences between FA3 and FlashInfer
- Common errors and troubleshooting

Overview
--------

LightLLM uses an offline calibration flow for FP8 KV quantization:

1. Run export mode to collect KV statistics and produce ``kv_cache_calib.json``.
2. Run inference mode with that file, and quantize KV into ``float8_e4m3fn`` storage.

Backend and Quantization Granularity
------------------------------------

Current behavior:

- ``fa3``: ``per_head`` scales (independent scale per head)
- ``flashinfer``: ``per_tensor`` scales (one scalar for K and one scalar for V)

Calibration files are backend-dependent:

- ``per_head`` files exported with ``fa3`` should be used with ``fa3`` inference.
- ``per_tensor`` files exported with ``flashinfer`` should be used with ``flashinfer`` inference.

Avoid mixing calibration files across different backends.

Step 1: Export Calibration File
--------------------------------

Export mode example (FA3):

.. code-block:: console

    $ python -m lightllm.server.api_server \
        --model_dir /path/to/model \
        --export_fp8kv_calibration \
        --llm_prefill_att_backend fa3 \
        --llm_decode_att_backend fa3 \
        --disable_cudagraph

Export mode example (FlashInfer):

.. code-block:: console

    $ python -m lightllm.server.api_server \
        --model_dir /path/to/model \
        --export_fp8kv_calibration \
        --llm_prefill_att_backend flashinfer \
        --llm_decode_att_backend flashinfer \
        --disable_cudagraph

Notes:

- Setting ``--export_fp8kv_calibration`` collects KV statistics during runtime.
- After calibration is completed, ``kv_cache_calib.json`` is written to the current working directory.
- Export mode requires ``--disable_cudagraph``, and ``--llm_kv_type`` should remain ``None``.
- The repository already provides calibration files for common models under ``test/advanced_config/``, which can be used directly or as references.

Use benchmark_qps.py for random-data calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Besides online traffic, you can use ``test/benchmark/service/benchmark_qps.py`` to generate random requests for calibration.

- By default, one calibration result is exported after around 4000 inferences are accumulated.
- In practice, you can run the following command twice to improve coverage stability.

Example command:

.. code-block:: console

    $ python test/benchmark/service/benchmark_qps.py --url http://127.0.0.1:8000/generate_stream --tokenizer_path ../Qwen3-30B-A3B --input_len 1000 --output_len 2000 --input_qps 10 --input_num 200 --range_ratio 0.9

Step 2: Start FP8 Inference with Calibration
---------------------------------------------

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
- Keep the inference backend consistent with the backend used during calibration export.

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

2. Error says ``--disable_cudagraph`` is required

    You are using ``--export_fp8kv_calibration``; this mode requires cudagraph disabled.

3. ``quant_type not match`` error

   Usually caused by backend/file mismatch (for example, using a ``per_head`` file with ``flashinfer``).

4. Abnormal quality after backend switch

   Re-export calibration using the target backend instead of reusing files across backends.
