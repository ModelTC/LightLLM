# DeepSeek-V4-Flash serving (run inside the lightllm container, repo mounted at /data/wanzihao/lightllm-ds4).
# Verified 2026-06-11: smoke + gsm8k pass with this configuration (prompt cache ENABLED, decode
# cudagraph ENABLED; gsm8k 100q/128: cold 0.960/112s, warm 0.970/23.5s with 100% cache hits —
# vs eager cold 0.970/141s, warm 0.960/50s; batch-1 decode 20.4ms/token vs 142ms eager).
#
# Required env/flags and why:
#   LOADWORKER=16      - parallel weight loading (~5x faster startup).
#   PYTHONPATH sglang  - _get_qkv / compressor reuse sglang.jit_kernel.dsv4 (fused_q_norm_rope, compress_old).
#   --batch_max_tokens 8192        - FlashMLA get_decoding_sched_meta rejects >8192 rows per call (probed: 8192 OK, 12288 fails).
#   decode cudagraph ENABLED - the v5 decode path is graph-safe: slot alloc/scatter in prep (outside
#     graph), forward is pure gathers, HOLD padding rows redirect to HOLD slots. CORRECTNESS NOTE:
#     FlashMLASchedMeta is lazily planned at first kernel call and written back onto the (shared)
#     decode att state; the capture warmup pass would bake a dummy-content plan into the graph
#     (gsm8k dropped to 0.74 with coherent-but-runaway generations). reset_sched_meta_for_capture()
#     in cuda_graph._capture_decode re-plans INSIDE the captured region so every replay re-plans.
#     DSV4 caps graph max_len_in_batch at 8192; longer decode batches fall back to eager.
#   --enable_prefill_cudagraph + --prefill_cudagraph_max_handle_token 2048 - graph-sandwich prefill:
#     graphs capture only the per-token dense ops; attention/compressor/indexer run eagerly between
#     graph segments (att_func), so host-side planning and .tolist() prep never enter capture. Only
#     cold prefills (prefix_total_token_num == 0, model gate) of <= 2048 new tokens replay; cache-hit
#     and large batched prefills stay eager. Buckets are padded with a HOLD tail request whose
#     attention output MUST be zeroed (infer_struct._dsv4_prefill_pad_q_len): pad rows read the
#     racing HOLD slot, and nondeterministic pad hiddens perturb real rows via MoE expert batching
#     (ulp-level, chaotically amplified ~1.9x/layer to O(1) by layer ~16 -> greedy token flips).
#     Residual caveat: padded-vs-unpadded expert-batch composition still shifts reductions by ulps,
#     same class as decode bucket padding; run-to-run determinism is anyway bounded by the fp4
#     marlin MoE kernel itself (probabilistic 1-ulp reduction-order noise measured eager-vs-eager).
#     Acceptance is therefore statistical (gsm8k parity), not bitwise.
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
  --enable_prefill_cudagraph \
  --prefill_cudagraph_max_handle_token 2048 \
  --port 8000
