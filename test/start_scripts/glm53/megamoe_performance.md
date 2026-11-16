# GLM-5.3-Flash MegaMoE performance on m35

Measured on November 16, 2026, using four H200 GPUs. Across the four tested workloads,
MegaMoE increased median output throughput by 14.5–20.1%, reduced time per output
token by 12.6–14.2%, and reduced time to first token by 9.8–21.7%.

| Input / output tokens | Concurrency | DeepEP output tok/s | MegaMoE output tok/s | Throughput gain | DeepEP TTFT ms | MegaMoE TTFT ms | DeepEP TPOT ms | MegaMoE TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 / 128 | 1 | 58.9 | 70.7 | 20.1% | 322.1 | 252.1 | 14.29 | 12.25 |
| 512 / 128 | 4 | 234.7 | 272.0 | 15.9% | 369.6 | 333.3 | 14.27 | 12.31 |
| 512 / 128 | 16 | 620.0 | 710.1 | 14.5% | 871.5 | 728.4 | 19.44 | 17.00 |
| 2048 / 128 | 4 | 177.2 | 210.9 | 19.0% | 1098.3 | 867.4 | 14.11 | 12.23 |

Values are medians of three trials. TTFT and TPOT use each trial's per-request
median. Output throughput includes prefill and equals the generated token count
divided by the full client trial duration. TPOT averages the intervals after the
first token. Every measured request generated exactly 128 tokens with EOS ignored.

## Workload and configuration

- Machine: m35; GPUs 0, 1, 3, 4; model `/mtc/models/GLM-5.3-Flash`.
- Source: `23e3a15f7da805e23dfcef6c419389c6923999b4` plus the MegaMoE changes.
  Exact source hashes and both runner versions are archived with the experiment.
- Both backends used EP4/DP4, FP8 experts, BF16 hidden/SSM states, Triton attention,
  batch/chunk capacity 256, request limit 16, token pool 8192, autotune level 0,
  decode CUDA graph batches 1–4, and `--linear_att_cache_size 1`.
- Baseline: `--ep_moe_backend auto` and `LIGHTLLM_ENABLE_SM90_FP8_MEGA_MOE=0`.
  Candidate: `--ep_moe_backend megamoe`. All ranks' actual buffer selection was verified.
- The required hybrid prompt cache stayed enabled. Distinct first 16 token IDs
  prevented full-page prefix hits. Both backends received the same seeded input
  sets, verified by hashes.
- The client reused `test/benchmark/service/benchmark_glm53_flash.py`'s streaming
  generator on the same host. Each case had two 32-token warmup waves and one
  128-token warmup wave, then three closed-loop trials with
  `max(4, 4 * concurrency)` requests each.
- Runtime: Torch 2.11.0+cu130. SM90 DeepGEMM reference:
  `dbbb0f4745af11c4c899ea7648888af687394ede`; setup is in [the guide](README_megamoe.md).

## Repeat ranges and validity

| Input tokens | Concurrency | DeepEP output tok/s range | MegaMoE output tok/s range | Measured requests per backend |
|---|---:|---:|---:|---:|
| 512 | 1 | 58.3–59.9 | 66.8–70.9 | 12 |
| 512 | 4 | 234.6–236.8 | 271.0–277.0 | 48 |
| 512 | 16 | 611.7–624.3 | 709.3–716.1 | 192 |
| 2048 | 4 | 161.4–177.3 | 210.1–212.1 | 48 |

These are observed three-trial ranges. The audit accepted 24 trials and 600
measured requests: exact lengths, matched input hashes, zero request errors,
zero retries, and no sampled compiler overlap or thermal events during measurement.
All recorded SM clocks in timed windows were 1980 MHz; temperatures were 35–54°C.

Preliminary data on GPUs 3, 4, 6, 7 was excluded because GPU 6 reached 86–88°C
with severe clock reduction during timed requests. Both backends were then
rerun on GPUs 0, 1, 3, 4. No fan, power, or clock settings were changed.
The valid baseline completed before a subsequent candidate startup encountered
NCCL-port `TIME_WAIT` validation. Only the candidate was resumed after a
port-availability wait; its launch settings and workload matched the valid baseline.

The comparison covers these four loads and a sequential baseline/candidate run.
Eight-GPU scaling, SLO-qualified capacity, other batch/attention configurations,
and model benchmark accuracy remain unmeasured.

## Output consistency and accuracy limits

A subsequent audit paired the saved greedy outputs by input hash. All 300 pairs
had the same first generated token. Complete 128-token sequences matched in
263 pairs and differed in 37:

| Input tokens | Concurrency | Exact 128-token matches / paired requests |
|---|---:|---:|
| 512 | 1 | 12 / 12 |
| 512 | 4 | 41 / 48 |
| 512 | 16 | 165 / 192 |
| 2048 | 4 | 45 / 48 |

These serving runs did not produce identical token sequences. Both backends use
FP8 expert weights, but intermediate quantization and rounding differ. This audit
does not separate backend effects from batch/scheduling variation. These synthetic performance
inputs have no labeled answers, and EOS was ignored to fix the output length.
The 37 differing sequences do not measure an accuracy loss. The distributed
kernel test's 0.0279–0.0302 relative RMSE is against its numerical reference;
it is neither a default-backend comparison nor a model accuracy percentage.
Task accuracy requires a separate matched labeled evaluation before making
an accuracy-preservation claim.

## Reproducible records

Directories under `/data6/exp_logs/runs/`:

- Combined results: `261116-193400-python3-tmp-glm53-megamoe-launch-performance-py-`
- Original valid baseline: `261116-192322-python3-tmp-glm53-megamoe-launch-performance-py`
- Excluded thermal run: `261116-190609-python3-tmp-glm53-megamoe-launch-performance-py`
- Saved-output audit: `261116-201149-python3-tmp-glm53-megamoe-output-audit-py`

The combined record includes `REPORT.md`, `comparison.json`,
`performance-cool-results/` with raw requests and server logs, launch manifests,
source/runtime receipts, GPU/compiler telemetry, `timed-gpu-audit.json`, both
benchmark runner versions, the preliminary exclusion audit, and cleanup evidence.
The dedicated container was stopped; test GPUs returned to idle and unrelated
GPU 2/5 allocations were preserved.
