NVFP4 KV Cache
==============

``--llm_kv_type nvfp4kv`` uses the vLLM/FlashInfer NVFP4 quantization recipe:
K/V use E2M1, with one dynamic E4M3 scale per token, KV head, and group of 16 features.
Global K/V scales are fixed at one, matching the fallback when a checkpoint does not
provide calibrated global scales. No calibration export is required, and
``--kv_quant_calibration_config_path`` must not be supplied.

.. code-block:: bash

    python -m lightllm.server.api_server \
      --model_dir /path/to/model --tp 2 --dp 1 \
      --data_type bfloat16 --llm_kv_type nvfp4kv \
      --llm_prefill_att_backend auto --llm_decode_att_backend auto

Q retains FP16/BF16 precision; both prefill and decode read quantized K/V.
Install the dependencies from the repository's ``requirements.txt``. On H200,
``auto`` prefers FA3 prefill and FlashInfer decode. Native FlashInfer supports
head dimensions 128 and 256. Other head dimensions, MTP block decode, noncausal
decode, and sliding-window layers use Triton. You can also explicitly select
``triton`` for both attention backends. Setting ``FLASHINFER_DISABLE_JIT`` disables
the additional small-tile JIT optimization.

Including scales, ordinary GQA KV storage occupies 9/32 of its BF16 size,
allowing approximately 3.56 times as many tokens in the same KV memory budget.
This ratio excludes attention workspace and other model allocations.

Supported configurations include Llama/Qwen GQA/MHA/MQA with head dimensions
16, 32, 64, 128, or 256, and full-attention layers in Qwen hybrid models.
CUDA Graphs, prefix caching, sliding windows, and MTP block decode are supported.
These ordinary GQA paths do not support ALiBi or attention sinks.
Standard DeepSeek/GLM MLA, GLM-5.3 Flash sparse MLA, and DeepSeek-V4 also support
NVFP4; see :doc:`kv_cache_quantization` for their commands and hardware requirements.
DeepSeek-V4 retains BF16 RoPE, FP8 indexers, and attention sinks. Its compressed
history, sliding windows, and indexer storage mean the ordinary GQA capacity ratio
does not apply to its complete KV storage. DeepSeek-V3.2 sparse MLA is not supported.
This is LightLLM's implementation of the recipe, not a claim about vLLM engine performance.

Accuracy was evaluated on all 1,319 GSM8K questions using Qwen2.5-14B-Instruct,
H200 TP2, and MTP=0. Under the same evaluation protocol, the repository's matching
metric was 92.19% for BF16 and 92.27% for NVFP4, a difference of +0.08 percentage
points (paired-test p=1.000). Strict matching was 91.05% and 90.75%, respectively,
a difference of -0.30 percentage points (p=0.755). Neither metric showed a
statistically significant difference; these results do not guarantee accuracy
on other models or datasets. Long greedy outputs can still differ after preemption
and recomputation, as also observed with BF16. Token-for-token output equality
across batch changes or preemption is not guaranteed.
