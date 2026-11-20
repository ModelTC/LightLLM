# GLM-5.3-Flash SM90 FP8 MegaMoE

Select fused communication and expert computation with
`--enable_ep_moe --expert_dtype fp8 --ep_moe_backend megamoe`.
This path follows [SGLang PR #29016](https://github.com/sgl-project/sglang/pull/29016).
The default `auto` backend keeps its existing selection.

Requirements:

- A single node with NVLink-connected SM90 GPUs and BF16 hidden states.
- All routed experts use `fp8w8a8-b128-deepgemm`.
- DeepEP and the usual LightLLM dependencies.
- The SM90 FP8 API from the SGLang DeepGEMM fork. The reference revision is
  `dbbb0f4745af11c4c899ea7648888af687394ede` (DeepGEMM PR #48, based on PR #36).
  It exposes `fp8_mega_moe`, `mega_moe_pre_dispatch_sm90`, and
  `get_symm_buffer_for_sm90_mega_moe`.
  Build its wheel with `bash build_sgl_deep_gemm.sh` against the serving
  runtime's Torch/CUDA. The distribution is named `sgl-deep-gemm`, and its
  import name is `deep_gemm`. The adapter also accepts the built package
  directory staged as `sgl_deep_gemm` on `PYTHONPATH`, which lets ordinary
  GEMMs retain their existing `deep_gemm` installation. Include the native
  extension, shared Python modules, and JIT headers from the built wheel.
  Do not put the fork's unbuilt source root on `PYTHONPATH`.

Redundant experts, online expert updates, RL mode, and microbatch overlap are unsupported.
The shared expert remains separate and runs with GLM's original clamp semantics.
MegaMoE interleaves the routed gate/up weights in their preallocated storage
after loading, leaving FP32 block scales unchanged. Those weights must then
remain on the MegaMoE path.

Both prefill and decode use one symmetric buffer, sized to the larger of the
prefill and decode dispatch capacities. Set
`NUM_MAX_DISPATCH_TOKENS_PER_RANK_PREFILL` to cover the largest local prefill batch.
Include speculative verification tokens and CUDA graph padding in
`NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE`. The kernel rejects
batches exceeding the buffer. Pre-dispatch clears unused routes on every call,
including when a rank has no tokens.

Example (adjust device assignment, capacities, ports, and checkpoint path):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_MAX_DISPATCH_TOKENS_PER_RANK_PREFILL=1024 \
exp -m "GLM53 MegaMoE: record revision, GPUs, flags and workload" \
python -m lightllm.server.api_server \
  --model_dir /models/GLM-5.3-Flash --trust_remote_code \
  --tp 4 --dp 4 --enable_ep_moe --expert_dtype fp8 \
  --ep_moe_backend megamoe --data_type bf16 \
  --port 8017 --mem_fraction 0.8 \
  --batch_max_tokens 1024 --chunked_prefill_size 1024 \
  --running_max_req_size 32 --max_req_total_len 8192
```

Run the distributed numerical check on idle GPUs through `exp`:

```bash
CUDA_VISIBLE_DEVICES=0,1 exp -m "GLM53 MegaMoE distributed correctness" \
torchrun --standalone --nproc-per-node=2 test/kernel/test_sm90_mega_moe.py
```

The check uses GLM's 288 experts, hidden size 4096, intermediate size 2048,
top-8 sigmoid routing, routed scaling factor 2.5, and SwiGLU limit 10.
It compares against an independent PyTorch reference with FP8 input and
intermediate rounding, checks shrinking/empty uneven batches, and replays
CUDA graphs. Synthetic expert validation does not establish model accuracy
or a serving speedup; those require checkpoint-based inference and matched
workloads against the default backend.

## Main-based port validation

The port is based directly on main `1233f83af085949622ee22153c4950718c0c197b`.
On m35 H200 GPUs 0 and 1, 51 focused regressions passed, covering the new adapter,
weight preparation, buffer allocation, startup restrictions, and existing GLM
and shared-expert behavior. The two-rank numerical test also passed at the GLM
dimensions above: nonempty relative RMSE was 0.0279–0.0302, shrinking and uneven
batches including an empty rank passed, and two CUDA graph replays matched eager
execution exactly. Repository-pinned Black and Flake8 passed for all 13 changed
Python files.

Records under `/data6/exp_logs/runs/`:

- Unit tests and JUnit: `261116-203800-python3-tmp-glm53-megamoe-main-validate-py-unit`
- Distributed GPU check: `261116-203855-python3-tmp-glm53-megamoe-main-validate-py-gpu`

Full-checkpoint serving, serving performance, and labeled model accuracy have
not been re-evaluated on the main-based port. The results below use the earlier
optimization-branch implementation.

## Reference validation on m35

The original implementation was tested on November 16, 2026, on `glm53_optim`
base `23e3a15f7da805e23dfcef6c419389c6923999b4`, with H200 GPUs and the
dependency revision above. These checkpoint and performance results precede the
port to main's split MoE modules:

- 41 focused unit/regression tests, including GLM shared-expert behavior and backend validation.
- The distributed numerical check at the default GLM dimensions on two GPUs.
  Relative RMSE was 0.0279–0.0302 against the reference; graph replay matched eager
  execution exactly. Shrinking batches and an empty rank passed.
- Full `/mtc/models/GLM-5.3-Flash` checkpoint serving with EP4/DP4 on GPUs 3, 4, 6, 7.
  All ranks selected `fp8xfp8` MegaMoE and captured decode graphs for batches 1–4.
  Five requests, including three concurrent requests and a repeated request,
  finished with `stop` and returned the expected answer values (4, Paris, 10).
  The smoke test used BF16 SSM state, batch capacity 256, and `--linear_att_cache_size 1`.

The one-entry CPU cache was necessary for this test environment: a fresh process
without model kernels independently reproduced a CUDA pinned-allocation error
for the default eight-entry, 544 MiB SSM cache; the 68 MiB allocation passed.
Responses included reasoning in `content`; this smoke test checks completed
generation and answer values, rather than response-format compliance.
Model benchmark accuracy remains unmeasured. A matched serving comparison against
the default backend is recorded in [the performance report](megamoe_performance.md).

Experiment records under `/data6/exp_logs/runs/`:

- Unit tests and JUnit: `261116-182916-python3-tmp-glm53-megamoe-run-py-cpu-unit-tests-`
- Distributed GPU check: `261116-182130-python3-tmp-glm53-megamoe-run-py-gpu`
- Checkpoint completions: `261116-184656-python3-tmp-glm53-megamoe-launch-checkpoint-py`
  (server log, responses, launch manifest, source hashes, runtime receipt, extension binary, and runners).
- Independent allocation probes: `261116-184452-python3-tmp-glm53-megamoe-pinned-probe-py`
  and `261116-184548-python3-tmp-glm53-megamoe-pinned-probe-py`.

The dedicated test container was stopped after validation. Unrelated GPU workloads were preserved.
