---
name: glm52-static-matrix-benchmark
description: Run and resume the four-node LightLLM GLM-5.2 static-inference prefill cache-hit matrix and constrained decode batch-size search, retain per-node raw logs, recover from CUDA OOM by cautiously lowering mem_fraction, render the requested Markdown result tables, and compute cost-analysis tables from measured prefill/decode throughput. Use for four-node GLM-5.2 long-context benchmark requests involving ctx/hit matrices, max_total_token_num, QPS/TPS/logical TPS, decode ITL limits, maximum batch size under latency constraints, or per-million-token cost estimates.
---

# GLM-5.2 static matrix benchmark

Run all commands from the LightLLM repository root. Use
`scripts/run_matrix.py`; do not transcribe benchmark numbers manually.

## Workflow

1. Check SSH access and eight idle GPUs on each default node: `10.119.17.190`,
   `10.119.17.202`, `10.119.17.179`, and `10.119.17.169`. Confirm the shared
   `${HOME}/models/GLM-5.2` and repository paths exist on every node.
2. Run `prefill`. This sends all six context lengths and five hit rates in
   one model load. Preserve its raw log and parsed JSON.
3. Run decode probes with `decode-probe`. Set `--max-batch-size` to the local
   per-DP cap; the reported `bs` is the globally aggregated batch size. Probe
   monotonically increasing caps. For each `(ctx, ITL limit)`, retain only
   successful rows whose measured `itl_ms <= limit`, then select the largest
   reported `bs`. A row above a limit is not eligible.
4. Continue probing until each selected point is bracketed by the next local
   batch size (or by the KV-capacity maximum). Use exact integer probes near a
   boundary; do not claim a sampled grid point is the maximum without a
   bracket.
5. Run `render` to create the two Markdown tables from saved measurements.
6. If cost analysis is requested, run `cost` after `render`. It reads the
   saved measurements, uses prefill `logical_tok/s` and decode `tok/s`, and
   emits one cost table for each decode ITL limit.

Default matrix:

- contexts: `16384,32768,65536,131072,204800,256000`
- prefill hits: `0,0.5,0.7,0.8,0.9`
- decode ITL limits: `20,33,50` ms
- decode output length: `600`

The display labels are `16k, 32k, 64k, 128k, 200k, 250k`. Do not run or
render 1M cases.

## Commands

```bash
python skills/glm52-static-matrix-benchmark/scripts/run_matrix.py prefill
python skills/glm52-static-matrix-benchmark/scripts/run_matrix.py decode-probe --max-batch-size 8
python skills/glm52-static-matrix-benchmark/scripts/run_matrix.py render
python skills/glm52-static-matrix-benchmark/scripts/run_matrix.py cost --monthly-rent 280000
```

For the fixed GLM-5.2 MTP scenario, add `--mtp` to every command. The runner
must inject exactly `--mtp_mode eagle_with_att`, `--mtp_step 2`,
`--mtp_draft_model_dir ~/models/GLM-5.2/`, and `--mtp_accept_rate 0.85`.
MTP output defaults to `benchmark_results/glm52-static-matrix-4n-mtp/`, and its
decode table includes the measured `accept` column.

Pass `--contexts` to a decode probe to test only selected contexts. Results
are resumable under `benchmark_results/glm52-static-matrix-4n/`; an existing
successful probe is reused unless `--force` is supplied. Every case retains
the rank-0 `.log` plus `.rank1.log`, `.rank2.log`, and `.rank3.log`.

The runner defaults to four 8-GPU nodes, global `tp=32`, global `dp=32`,
`expert_dtype=fp8`, `batch_max_tokens=4096`, and
`graph_max_batch_size=200` for H100. The historical 8192-token default is for
B300 and must be explicitly requested there. Override topology with
`--node-hosts`, `--gpus-per-node`, `--nccl-host`, or `--nccl-port`. Use
`--expert-dtype fp4` only on SM100 GPUs.

Prefill defaults to `--chunked_prefill_sizes 2048` and
`--enable_prefill_microbatch_overlap`. The runner also passes
`--enable_tpsp_mix_mode`, which the normal service entry enables implicitly
for microbatch overlap but the static benchmark entry requires explicitly.
Override the chunk size with `--prefill-chunk-size`; keep the resulting local
prefill batch size even so it can split into two microbatches. These prefill
flags must not be added to `decode-probe`.

Pass `--nccl-port` with a confirmed free port when another benchmark process
is using the default rendezvous port. Keep probes serialized.

## Cost analysis

Run cost analysis only after the prefill and decode tables are complete enough
for the requested SLA. The deterministic command is:

```bash
python skills/glm52-static-matrix-benchmark/scripts/run_matrix.py cost --mtp --monthly-rent 280000
```

For B300, default to `--monthly-rent 280000` unless the user gives another
price. The output defaults to `cost_analysis.md` in the benchmark output
directory, e.g. `benchmark_results/glm52-static-matrix-mtp/cost_analysis.md`.

Use this formula family:

```text
machine_cost_per_second = monthly_rent / 30 / 24 / 3600
input_cost_yuan_per_M_tokens = machine_cost_per_second / prefill_logical_tok_s * 10^6
output_cost_yuan_per_M_tokens = machine_cost_per_second / decode_tok_s * 10^6
```

For cache-hit input costs, use the measured `logical_tok/s` at that hit rate;
do not linearly discount the cache-miss cost. For decode output costs, select
the same maximum-`bs` row used in the decode table for each ITL limit
(`<=20`, `<=33`, `<=50`). The cost report must include three Markdown tables,
one per ITL limit, with the columns:

```text
ctx
输入cache miss成本 元/M tokens
输出成本 元/M tokens
输入@50% hit平均成本 元/M logical tokens
输入@70% hit平均成本 元/M logical tokens
输入@80% hit平均成本 元/M logical tokens
输入@90% hit平均成本 元/M logical tokens
```

## Invariants

- Add `--benchmark prefill` or `--benchmark decode`; never mix stages in a
  measurement run.
- Use the repository root for `PYTHONPATH`. The lowercase path from an old
  command may not exist on case-sensitive filesystems.
- Launch all node ranks through the runner; do not start four independent
  rank-0 static benchmarks.
- Keep all model, TP/DP, KV, quantization, EP, expert dtype, token budget, and
  output-length settings from the selected four-node baseline command.
- Treat `max_total_token_num` printed by the benchmark as authoritative.
- For MTP CUDA OOM, first retry the same probe with `batch_max_tokens=4096`.
  Only if that still OOMs, lower `mem_fraction` by `0.02`. For non-MTP CUDA
  OOM, lower `mem_fraction` by `0.02`. Record the actual values in the run
  manifest; do not silently compare rows produced with different settings.
- Never invent, interpolate, or extrapolate a benchmark value.
- Keep raw stdout/stderr even for failed attempts.
