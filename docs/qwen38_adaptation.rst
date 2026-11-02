Qwen3.8-Flash-Next
==================

Qwen3.8-Flash-Next checkpoints use the ``qwen4_exp`` model type. The model
implementation combines gated residuals, n-gram embeddings, QSA and GDN
attention, with dedicated multimodal, MTP and cache-state integration.
Reusable operations live in ``lightllm/common/gated_residual``,
``lightllm/common/ngram_embedding`` and ``lightllm/common/qsa``.

CPU embedding offload is disabled by default. The optional QSA TopK selector
is enabled by the model only for its qualified H100/Torch/Triton geometry;
other configurations use Torch. Shared-prefix admission is limited to the
model's single-node, DP1 normal serving mode with one-page state blocks.

Validation
----------

Historical H100 measurements and their configuration are recorded in
``qwen38_h100_performance.rst``. They cover TP8 BF16, FP32 SSM states,
fixed MTP2 and one-page linear-attention state blocks. These results do not
qualify PD/cross-DP serving or other cache block configurations.

The four-page-block pressure-recovery configuration has unresolved text
differences. Low-load latency still trails SGLang in some metrics. Archived
HC and sampling experiments are not part of the runtime implementation.

Rebase to upstream, 2026-11-03
------------------------------

The implementation now uses AttentionStateManager. QSA keys and positions are
owned by the model's GQA State and participate in token copies and CPU/PD/DP
transfer layouts. Lexical histories use a separate bounded State for reset,
prefix snapshots, CPU/PD/DP transfers and accepted MTP candidate commits.
The removed request/cache manager classes are not restored.

Model and draft registration use the upstream lazy registry. HTTP slot waiting
preserves atomic group allocation and current PD timeouts. The GDN MTP path uses
the upstream recurrent kernel and its explicit run configuration.

The historical H100 results above predate this interface migration. They do not
validate the rebased runtime. GPU generation, complete PD transfer and performance
must be rechecked before deploying a new image of this revision.

CPU validation after migration: 1022 checks passed and 930 GPU cases were
skipped. The DeepSeek V4 integration module could not be collected because
FlashMLA is absent. Two upstream State identity cases require CUDA-backed
pinned memory; both failures reproduce on unmodified upstream/main with CUDA
hidden. No GPU serving, PD end-to-end or performance result is claimed here.
