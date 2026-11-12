.. _tutorial/fp8_kv_quantization_en:

KV Cache Quantization Guide
===========================

Select a KV cache quantization mode with ``--llm_kv_type``.
Prepare the environment using the :doc:`installation guide <../getting_started/installation>`.

Choose a Mode
-------------

.. list-table::
   :header-rows: 1

   * - Mode
     - Use case
     - Calibration file
   * - ``fp8kv_sph``
     - Static per-head quantization; requires a Hopper GPU
     - ``per_head``, including Q calibration data
   * - ``fp8kv_spt``
     - Static per-tensor quantization; requires FlashInfer
     - ``per_tensor``
   * - ``fp8kv_dsa``
     - DeepSeek-V3.2 and GLM-5.3 Flash
     - Not required
   * - ``nvfp4kv``
     - Dynamic NVFP4; see MLA usage below
     - Not required

Static FP8 KV
-------------

Prepare a ``kv_cache_calib.json`` compatible with your model and quantization mode,
then pass its path at startup. Export one with
`LightCompress <https://github.com/ModelTC/LightCompress>`_ or use an existing compatible file.

Per-head quantization:

.. code-block:: console

    $ python -m lightllm.server.api_server \
        --model_dir /path/to/model \
        --llm_kv_type fp8kv_sph \
        --kv_quant_calibration_config_path /path/to/per_head/kv_cache_calib.json

Per-tensor quantization:

.. code-block:: console

    $ python -m lightllm.server.api_server \
        --model_dir /path/to/model \
        --llm_kv_type fp8kv_spt \
        --kv_quant_calibration_config_path /path/to/per_tensor/kv_cache_calib.json

Both modes select the attention backend automatically.
For multiple GPUs, add ``--tp <gpu_count>`` and provide one complete calibration file.

Dynamic FP8 KV
--------------

For DeepSeek-V3.2 and GLM-5.3 Flash, use ``--llm_kv_type fp8kv_dsa`` without a calibration file.
A compatible FlashMLA installation is required. You can build an image from this repository's
Dockerfile as described in the installation guide. When installing from source, install
FlashMLA separately; ``requirements.txt`` alone is insufficient.

The following example uses GLM-5.3 Flash, which requires ``bfloat16`` and supports only
``fp8kv_dsa`` for FP8 KV quantization:

.. code-block:: console

    $ python -m lightllm.server.api_server \
        --model_dir /path/to/GLM-5.3-Flash --tp 4 \
        --data_type bfloat16 --llm_kv_type fp8kv_dsa

NVFP4 MLA KV
------------

Use ``--llm_kv_type nvfp4kv`` for standard DeepSeek/GLM MLA or GLM-5.3 Flash sparse MLA.
Latents are dynamically quantized to E2M1 with one E4M3 scale per 16 elements;
no calibration file is required. Q retains the computation precision, and MLA with
RoPE retains its original RoPE precision. Standard MLA uses the ``triton`` backend.

GLM-5.3 Flash uses BF16 computation and retains its K-pool indexing and linear attention state:

.. code-block:: console

    $ python -m lightllm.server.api_server \
        --model_dir /path/to/GLM-5.3-Flash --disable_vision --tp 4 \
        --data_type bfloat16 --llm_kv_type nvfp4kv --page_size 16 \
        --chunked_prefill_size 4096

The GLM NVFP4 attention path requires a Hopper GPU and compatible FlashMLA and FA3
installations; follow the installation guide. KV remains compressed in storage;
attention uses temporary BF16 workspace as needed. Actual token capacity depends on
weights, runtime allocations, and the GPU memory budget.

GLM KV pages must be a multiple of four tokens. The default ``--page_size 1`` is
automatically adjusted to four; you can explicitly select a larger value such as 16.
CPU cache and P/D transfer pages must also be aligned to four tokens.
GLM native MTP uses the same NVFP4 encoding as the main model.

Disaggregated P/D Deployment
----------------------------

For dynamic FP8 KV in a disaggregated deployment, set ``--llm_kv_type fp8kv_dsa``
on both Prefill and Decode services.
For GLM-5.3 Flash on a single host with 8 GPUs, the following script uses 4 GPUs per service:

.. code-block:: console

    $ LLM_KV_TYPE=fp8kv_dsa bash test/start_scripts/glm53/glm53_pd_1p1d.sh \
        8088 <node_ip> /path/to/GLM-5.3-Flash

Replace ``<node_ip>`` with a reachable IP address of the host. The script configures the
quantization mode and transfer page size for both services. When starting services manually,
use the same quantization mode and ``--pd_kv_page_size`` on both sides.

Troubleshooting
---------------

- **Missing calibration file or calibration type mismatch:** Static modes require
  ``--kv_quant_calibration_config_path`` with a file matching the model and mode.
  Files for ``fp8kv_sph`` must also include Q calibration data.
- **Missing dependencies:** Follow the current installation guide. Dynamic mode also
  requires a compatible FlashMLA installation.
- **Insufficient P/D transfer page capacity:** Increase ``--pd_kv_page_size`` on both
  services and keep it a multiple of ``--page_size``.
