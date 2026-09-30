# ReplaySSM implementation and validation

ReplaySSM is opt-in: add `--enable_replayssm` to an existing Qwen3-Next or
Qwen3.5 command. `--replayssm_cache_len` accepts 16 (default), 32, or 64 and must
cover `mtp_step + 1`. Existing scheduling, convolution, cache-page formats,
prefill backends, and acceptance policy remain the owners of their current work.

## Runtime paths

| Recurrent state | Ordinary decode | Speculative verification |
| --- | --- | --- |
| GDN FP32 | Deferred checkpoint + accepted update history | Same representation; commit advances accepted prefix |
| GDN BF16 | Existing recurrent kernel | Compact inputs; replay accepted prefix with per-token rounding |
| GLM KDA, after model integration | Existing recurrent kernel | Compact inputs with bounded per-key gate |

FP32 history stores normalized keys, deltas, and cumulative log decay. Two
alternating history regions prevent cross-CTA overwrite races during folding.
Metadata is shared across layers; HOLD requests never acquire history. The verify
kernel processes candidates sequentially without writing full candidate states.
It does **not** implement SGLang's parallel triangular-solve verify formulation.

Before prefill reentry, CPU checkpoint saves, or PD state export, pending accepted
history is folded into canonical state. Restore/reuse clears history metadata.
CPU and PD snapshots select the accepted convolution window independently of the
recurrent cursor. Compact commit already leaves canonical recurrent state.

BF16 uses compact recomputation because deferring rounded state updates changes
the recurrence. Ordinary BF16/KDA decode keeps the existing kernel; compact replay
is useful for avoiding speculative full-state snapshots, not as a blanket
replacement for ordinary recurrence.

## GLM-5.3-Flash

The compact cache already supports KDA's per-key gate, but model routing is left
to [PR #1575](https://github.com/ModelTC/LightLLM/pull/1575). The two changes were
tested together at `7eef17bb360c9b454ed5707ea1175cafed8446dd`: kernel/MTP/cache/PD
tests and fixed/dynamic KDA adapter tests passed. Full GLM serving and multi-node
PD remain unvalidated, so this change does not advertise GLM support on main.

## Exact fold on m39

The GDN path separates output reconstruction from checkpoint maintenance.
Normalized keys and deltas are used only for verify outputs; flush and cache/PD
materialization replay accepted raw BF16 keys/values plus FP32 gates/betas into
the FP32 checkpoint. Approximate output records therefore never become the next
persistent state.

Verify uses the sequential output-only kernel. A minimum 16-token matrix tile
was rejected: it measured 0.15--0.17x of native at width 4 and 0.36--0.47x at
width 16 on the 48-layer reference shape. On an H200 in m39, width 4 with three
accepted tokens measured 1.06x/1.09x/1.16x at batches 32/68/96. Replay state and
history occupied about 44% of the native speculative SSM snapshot allocation.
Sources: `260920-165210-*` (rejected matrix-only result) and `260920-165530-*`
(sequential result); the width-16 matrix rejection is `260920-170004-*`.
Sequential width 16 improved to 0.79x but still regressed (`260920-170210-*`).
These are kernel measurements, not end-to-end serving gains.

## Measurements on m33

Hardware: H200; PyTorch 2.11.0+cu130, Triton 3.6. All measurements are recorded with
`exp -m`; IDs below identify `/root/experiments/runs` inside `lightllm-replayssm`
and exported copies under the m33 user's `~/experiments/runs`.

Kernel results include metadata, accepted-prefix commit, and recurring folds.
Shape: K=V=128, 16 query heads, 32 value heads, batch 256, capacity 16.

| Path | Verify width | Baseline ms | Replay ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| FP32 deferred | 1 | 0.35152 | 0.24927 | 1.41x |
| FP32 deferred | 3 | 0.85617 | 0.51800 | 1.65x |

Source: `260918-093117-*` (BV64/warps2).
Earlier BF16 compact measurements (`260918-094913-*`, 2.64–3.51x) used BV32.
Repeated comparisons against the actual native BV8 default exposed a rounding
difference. Final GDN compact now uses BV8 and matches native output/state over
32 acceptance cycles in the test. **The earlier compact timings are superseded**;
Final compact was remeasured on an idle H200 on September 20. With width3,
all candidates accepted, and one layer with eight value heads, speedups at
batches 1/4/16/32/68 were 0.90/0.95/0.98/1.12/1.28x (`260920-013240-*`).
A 48-layer chain with one final cross-layer commit, four key heads and twelve
value heads per layer, measured 1.54/1.28/1.31/1.50/1.56x at those batches
(`260920-013503-*`). The latter uses the local public Qwen3.5-27B configuration
divided by TP4 as a reference shape; it is not full-model serving. Both use CUDA
Graph timing and omit projections, full attention, communication, draft, sampling,
scheduling, and PD transfer. Single-layer timing does not predict the multilayer
chain, and neither establishes an end-to-end gain.
At width4 with three accepted tokens, the same 48-layer shape measured
3.512/2.023 ms (native/compact) at batch32, 7.514/4.007 ms at batch68, and
10.770/5.588 ms at batch96 (`260920-015547-*`). These H200 measurements support
testing MTP3 on the H100 deployment, but do not predict its service OTPS.
The speculative baseline uses the existing default BV8/warps1 configuration,
not an exhaustive baseline autotune. Small batches can regress: FP32 batch1 was
0.42x for ordinary decode, 0.49x for width3. These are kernel results, not service
speedups or proof of peak performance. Opt-in remains appropriate.

At width3/batch256, FP32 state plus history was 809,505,796 bytes versus
1,616,904,192 bytes baseline. BF16 compact used 282,313,472 versus 808,452,096 bytes.
These totals exclude unrelated model/KV memory.

Qwen3.5-0.8B serving exercised ordinary decode, dynamic MTP, and repeated prefix
cache hits (512 cached tokens). Four of five deterministic prompts matched all
128 tokens. One diverged after token57 for deferred ordinary decode and token61
for dynamic MTP, so bitwise generation parity is **not** claimed. Reassociation
changes numerical results even when recurrence tests pass.

The existing GSM8K script, first100 test questions after five few-shot examples,
scored 31/100 for baseline, FP32 deferred ReplaySSM, and BF16 compact dynamic MTP. Sources:
`260918-095130-*`, `260919-141406-*`, and `260919-141716-*`. This is a small accuracy smoke test.
The matching native BF16 dynamic-MTP baseline scored 32/100
(`260919-141956-*`). Several answers differed even in serial repeated requests
(`260919-142125-*`), prompting the BV8 correction above; initial BF16 serving
results describe the superseded BV32 kernel. The later runs shared GPUs with
other workloads; their latency is not comparable.
The corrected BV8 compact run scored 30/100 (`260919-142538-*`), and four of
five serial probes matched native text (`260919-142626-*`). One probe still
differed, so full serving numerical equivalence remains under investigation.
Repeating the native concurrent baseline scored 33/100 (`260919-142927-*`),
showing that its 100-question score itself varies with execution conditions.
A diagnostic service ran native verification alongside compact GDN on identical
live inputs, then checked accepted state after commit. Across ten requests on
these five probes it logged no output/state mismatches (`260919-143150-*` client
run; diagnostic server source/log archived with the experiment). This localizes
the remaining service differences outside the tested recurrence operations;
it does not establish whole-model numerical parity.

## Reproduce

Run on a CUDA machine from the repository root:

```sh
python -m pytest -q unit_tests/common/basemodel/triton_kernel/linear_att/test_replayssm.py
PYTHONPATH=. exp -m 'ReplaySSM deferred kernels' python test/benchmark/kernels/benchmark_replayssm.py
PYTHONPATH=. exp -m 'ReplaySSM compact kernels' python test/benchmark/kernels/benchmark_replayssm.py --compact --state-dtype bfloat16 --widths 3 5
PYTHONPATH=. exp -m 'ReplaySSM serving parity' python test/benchmark/service/compare_replayssm.py --baseline-port 18932 --replay-port 18934
```

The main runtime suite passed 65 tests (two native KDA comparisons skipped on
main); all four extended native comparisons passed with the PR kernel injected.
Tests cover accepted prefixes, irregular verify groups, graph replay, alternating
ring folds, materialization, request reuse, CPU checkpoints, PD page helpers, and
compact/native recurrence comparisons. KDA native comparisons skip on main and
run against the PR's real kernel when it is present. The initial single-window KDA case matched BF16 exactly and had FP32 state error
at most 7.45e-9. Expanded changing-input tests exposed rounding differences across
repeated commits; KDA therefore uses explicit FP32/BF16 error tolerances and does
not promise bitwise equivalence. GDN compact retains exact native comparison.

Remaining qualification: larger task accuracy samples, isolated end-to-end
throughput, tuned-baseline comparisons, full GLM serving, and multi-node PD.

Algorithm references: [ReplaySSM](https://dao-lab.ai/blog/2026/replayssm/) and
[SGLang](https://github.com/sgl-project/sglang/tree/bbfcda48cebfbde2ed4be4e8273eb593c764e638).
