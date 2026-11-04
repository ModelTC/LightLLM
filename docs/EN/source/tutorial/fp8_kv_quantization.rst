.. _tutorial/fp8_kv_quantization_en:

FP8 KV Quantization Guide
=========================

Select an FP8 KV cache quantization mode with ``--llm_kv_type``.
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
