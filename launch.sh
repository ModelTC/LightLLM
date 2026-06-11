# DeepSeek-V4-Flash serving (run inside the lightllm container, repo mounted at /data/wanzihao/lightllm-ds4).
# Verified 2026-06-11: smoke + gsm8k pass with this configuration (prompt cache ENABLED, decode
# cudagraph ENABLED; gsm8k 100q/128: cold 0.960/112s, warm 0.970/23.5s with 100% cache hits —
# vs eager cold 0.970/141s, warm 0.960/50s; batch-1 decode 20.4ms/token vs 142ms eager).
#
# Required env/flags and why:
#   LOADWORKER=16      - parallel weight loading (~5x faster startup).
#   Optional sizing knobs (defaults shown): LIGHTLLM_DSV4_SWA_FULL_TOKENS_RATIO=0.1 (swa pool floor
#   as a fraction of full tokens; raise for long-prompt x high-parallel workloads),
#   LIGHTLLM_DSV4_PROFILE_MAX_FULL_TOKENS=1500000 (auto-profile cap on max_total_token_num).
#   PYTHONPATH sglang  - _get_qkv / compressor reuse sglang.jit_kernel.dsv4 (fused_q_norm_rope, compress_old).
#   --batch_max_tokens 8192        - FlashMLA get_decoding_sched_meta rejects >8192 rows per call (probed: 8192 OK, 12288 fails).
#   decode cudagraph ENABLED - the v5 decode path is graph-safe: slot alloc/scatter in prep (outside
#     graph), forward is pure gathers, HOLD padding rows redirect to HOLD slots. CORRECTNESS NOTE:
#     FlashMLASchedMeta is lazily planned at first kernel call and written back onto the (shared)
#     decode att state; the capture warmup pass would bake a dummy-content plan into the graph
#     (gsm8k dropped to 0.74 with coherent-but-runaway generations). reset_sched_meta_for_capture()
#     in cuda_graph._capture_decode re-plans INSIDE the captured region so every replay re-plans.
#     DSV4 caps graph max_len_in_batch at 8192; longer decode batches fall back to eager.
#   --disable_flashinfer_allreduce - flashinfer cuda_ipc resolves libcudart to tilelang's stub (undefined cudaDeviceReset); symm-mem allreduce is used instead.
#
# One-time container setup already applied (survives until container rebuild):
#   pip install ipython   (sglang import dependency)
#   site-packages/vllm: layers/mhc.py + kernels/mhc/ + _tilelang_ops.py overlaid from /data/wanzihao/vllm (mhc_pre_tilelang ops; original kept at layers/mhc.py.bak)
#
# original: python -m lightllm.server.api_server --model_dir /data/models/DeepSeek-V4-Flash --tp 4 --enable_prefill_cudagraph

# repo root = this script's directory, so the same file works in the main tree and in worktrees
# (a hardcoded tree path here once made a worktree launch silently serve main-tree code).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOADWORKER=16 \
PYTHONPATH="${REPO_DIR}":/data/wanzihao/sglang/python \
python -m lightllm.server.api_server \
  --model_dir /data/models/DeepSeek-V4-Flash \
  --tp 4 \
  --batch_max_tokens 8192 \
  --disable_flashinfer_allreduce \
  --port 8000
