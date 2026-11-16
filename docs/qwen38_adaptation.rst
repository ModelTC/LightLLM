Qwen3.8-Flash-Next
==================

Qwen3.8-Flash-Next checkpoints use the ``qwen4_exp`` model type;
text-only checkpoints use ``qwen4_exp_text``. The model
implementation combines gated residuals, n-gram embeddings, QSA and GDN
attention, with dedicated multimodal, MTP and cache-state integration.
Reusable operations live in ``lightllm/common/gated_residual``,
``lightllm/common/ngram_embedding`` and ``lightllm/common/qsa``.

CPU embedding offload is disabled by default; enable it with
``--offload_ngram_embedding``. Shared-prefix admission is limited to the
model's single-node, DP1 normal serving mode with one-page state blocks.

QSA supports BF16 KV storage and ``--llm_kv_type fp8kv_spt``. The latter
stores main KV and pooled indexer keys as FP8 e4m3 with unit scales; raw
indexer keys and rotary inputs retain the model dtype. Calibration files
are not supported for this QSA format. With ``--page_size 4``, pooled
indexer keys use one entry per compression group and unfinished groups
use request-local ring storage that includes MTP candidate capacity.

``--enable_ep_moe`` supports unquantized BF16 experts through DeepEP and
DeepGEMM. It does not require ``--expert_dtype fp8``. The BF16 path supports
ordinary prefill/decode and CUDA Graphs; split dispatch/GEMM interfaces
used by microbatch overlap still require quantized experts.

GDN supports ``--ssm_state_mode replay`` with FP32 SSM state and a bounded
replay history. It uses the upstream GDN state lifecycle: decode advances
the verification window, and MTP acceptance trims rejected tokens using
each request's verification width. Cache pages store prefill Conv/SSM state;
Qwen lexical state participates in the same acceptance call.

FlashInfer 0.7.2 or later provides an optional GDN prefill backend through
``--llm_prefill_att_backend auto flashinfer``. It accepts the existing KV
state layout through explicit transposes and preserves zero-length states.
Prefill CUDA Graphs reuse the captured GDN subgraph when the request count
matches; other request counts use the current batch state outside the graph.
QSA prefill replays its captured subgraph when the current visible context
fits the captured bound, including batches with a different request count.
Longer contexts use eager QSA attention with the current batch metadata.
The default linear prefill backend selection is unchanged.

Prefill graphs require workspace in addition to the KV cache. The evaluated
TP8 profile on eight H100 80GB GPUs uses
``--enable_prefill_cudagraph --prefill_cudagraph_max_handle_token 1024``
with ``--max_total_token_num 2097152``, 64 request slots and a 16K context
limit. Larger prefill batches retain the eager path. The explicit KV limit
reduces spare prefix-cache capacity; it is a deployment setting, not a
model requirement.

MTP projects draft logits only for accepted-tail rows while retaining all
hidden rows and KV writes. Its graph cache keys include both input and output
capacity, and all supported capacities are captured during startup before
target hidden states can occupy the shared graph pool. Ordered TP reduction
retains rank-ordered FP64 accumulation; the H100 peer path synchronizes
readers before reusing its shared buffer.

On H100, the TP4/DP1 and TP8/DP1 profiles with BF16 weights, FP8 QSA,
FP32 ReplaySSM with a 16-token precomputed history and fixed MTP2, MTP3
or MTP4 use a replicated small-batch decode path. Complete verification
groups of up to 64 rows retain the complete residual on each TP rank while
experts remain distributed across the TP ranks. Graph capacities and padding
follow the verification width; groups that exceed 64 rows use the existing
TP/SP alignment. Each layer reuses a bounded 64-row expert workspace across
sequential graph replays. Shared-expert gating is combined with the
routed-expert sum while preserving BF16 intermediate rounding. Larger batches,
unsupported expert layouts, quantized weights, microbatch overlap and
routed-expert collection use the existing path.

Within the TP8/MTP3 profile, QSA decode with up to 128 rows packs the attention
and indexer projections into one GEMM; prefill and draft execution retain
separate projections. TP8/MTP3 also fixes the QSA split count and ReplaySSM
kernel configuration across decode batch sizes. TP4, MTP2 and MTP4 retain
their existing QSA and ReplaySSM kernel settings. These optimizations do
not establish end-to-end deterministic generation.

For gated residuals with up to 64 rows, width at least 8192 and projection
rank at most 512, the down projection and injection gates share a kernel
launch. Their reductions and intermediate rounding remain separate.
This uses the existing weights and split workspace. Larger batches and
FP32 inputs retain separate gate execution. The projection kernels take
the row count at runtime to avoid recompilation for each prefill batch size.

Kernel microbenchmarks are in ``test/benchmark/qwen38/``. Numerical and
cache-state regression tests are in ``unit_tests/models/qwen4_exp/`` and
the corresponding ``unit_tests/common/`` directories.
