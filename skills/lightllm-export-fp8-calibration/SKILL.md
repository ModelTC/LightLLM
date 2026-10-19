---
name: lightllm-export-fp8-calibration
description: Create and verify a fresh LightLLM FP8 KV/QKV calibration export for a specified main model and DSpark draft model.
---

# Export a fresh LightLLM FP8 calibration

Use this skill when the user needs a new FP8 KV or QKV calibration JSON for a
specified main model and DSpark draft. Use the repository's
`tools/calibrate_fp8kv.py`; do not derive a result by renaming, copying,
merging, or extending scales from a previous JSON.

This workflow creates a new artifact and its evidence. It does not use an old
calibration JSON in inference as a substitute for collection.

## Inputs and baseline

Establish the absolute LightLLM repository, Python environment, main-model
path, DSpark draft path, output parent, GPU allocation, and any authorized
deviation. Pass a chat template only if the actual model requires it.

Unless the user overrides an item, use and record this baseline:

| Setting | Value |
| --- | --- |
| calibration target | `qkv` |
| samples / input source / seed | 100 / random token IDs / 42 |
| maximum input / generated tokens | 1024 / 2000 |
| concurrency | 16 |
| topology / run mode | single-node, TP 4, DP 1, normal |
| attention backends | FA3 prefill and decode |
| draft | DSpark |

When the user explicitly overrides a baseline value, derive the command,
lifecycle request count, GPU check, and acceptance checks from the approved
values recorded in `run_manifest.json`. The expected generated-token total is
`num_samples × max_new_tokens`. Call a run a complete 2026-09-15 baseline only
when it has no relevant baseline deviation; a valid authorized override is
complete for its recorded configuration, not for that historical baseline.

`max_input_tokens=1024` is the current tool default selected for this baseline;
it was not recorded in the historical 2026-09-15 report. Random inputs come
from the complete legal candidate pool: the tool sorts and de-duplicates
vocabulary IDs, excludes special and out-of-model-vocabulary IDs, then uses
`random.Random(42)`, `randint(1, 1024)`, and `choice`. This is random sampling
with replacement, not natural-language or chat-template prompting, and it does
not guarantee that every candidate token appears. Do not replace it with full
vocabulary traversal or balanced sampling: that would diverge from the baseline.
Different tokenizers can change sampled lengths, total tokens, and the hash.

Read [references/baseline-20260915.md](references/baseline-20260915.md) only
when comparison with the historical run is needed. It is evidence for
provenance, not a target a new model must reproduce.

## Never reuse old scales

For a new main/draft pair, collect a wholly new JSON. In particular:

- do not pass an old `--kv_quant_calibration_config_path` to collection;
- do not use `--calibration_target q` to add Q to an old KV JSON;
- do not copy Q scales, KV scales, or a historical layer layout;
- do not pass `--overwrite` by default.

Create a unique absolute output directory and file name with model label, UTC
timestamp, and random suffix. It must not exist. Retain JSON, report, tool run
directory, `service.log`, start arguments, full argv, and `run_manifest.json`.
Never delete a report after export.

## Decide the calibration target

Inspect the target checkout before selecting the target. On the current branch,
`fp8_static_per_head_quant_mem_manager.py` requires `q_calibration`; therefore
the default is `--calibration_target qkv`. The 2026-09-15 artifact was KV-only;
QKV is the current compatibility extension while retaining its request lifecycle
and sampling budget.

Use `--calibration_target kv` only when the user expressly requests historical
KV-only behavior or the target checkout's loader does not require Q. If the
current loader requires Q, report that it cannot directly load KV-only output.
If the tool/check-out lacks QKV export while the loader requires Q, stop and
report the incompatibility. Never fabricate Q scales.

Read actual runtime layout. Do not assume target=16, draft=3, or MTP step=3.
For DSpark, derive the draft step from the checkpoint `block_size` in the
current implementation; if a version needs an explicit step, use that model's
value and record it. Context must cover the selected maximum input and generated
tokens, plus runtime draft overhead.

## Preflight

Perform only checks relevant to this run:

1. Validate main/draft paths and architecture compatibility, export
   `MemoryManager`, dependencies, FA3 availability, tokenizer, context limit,
   and the actual target/draft layout.
2. Confirm the selected TP number of compatible GPUs with read-only
   `nvidia-smi` (four for the baseline); confirm selected ports are free with
   read-only `ss -ltnp`.
3. Confirm the output does not exist and can retain the JSON, report, logs, and
   manifest.
4. Inspect whether the target loader requires `q_calibration`.

Do not kill other services, use global `pkill`, or silently reduce TP,
concurrency, samples, or output length. Report insufficient GPUs or an
incompatible environment; change the budget only with user authorization.

## Prepare and record the command

Use absolute paths and variables prefixed `CAL_`. Let the tool choose its own
free HTTP port. Check a free NCCL port before setting `CAL_NCCL_PORT`. Adapt
only genuine model requirements such as chat template, data type, or quantized
weight support; do not hard-code historical paths.

This is the baseline template; substitute only explicitly approved manifest
values when a user override applies.

```bash
cd "$CAL_REPO"
python tools/calibrate_fp8kv.py \
  --calibration_target qkv \
  --num_samples 100 --max_input_tokens 1024 --max_new_tokens 2000 \
  --concurrency 16 --seed 42 \
  --startup_timeout 600 --request_timeout 600 --drain_timeout 120 \
  --output "$CAL_OUTPUT" \
  -- \
  --model_dir "$CAL_MODEL" \
  --mtp_mode dspark --mtp_draft_model_dir "$CAL_DRAFT" \
  --tp 4 --dp 1 --nnodes 1 --run_mode normal \
  --llm_prefill_att_backend fa3 --llm_decode_att_backend fa3 \
  --nccl_port "$CAL_NCCL_PORT"
```

For an authorized KV-only run, replace only `qkv` with `kv` after documenting
loader compatibility. Record shell-safe argv plus repository revision and dirty
state; exact sampling parameters including maximum input; tokenizer, main, and
draft IDs; Python/GPU environment; actual TP/DP; and actual target/draft layout
in `run_manifest.json`.

## Understand the lifecycle

The tool owns a loopback-only service process group. It:

1. starts an unquantized-KV service with required CUDA-graph/cache settings;
2. waits for health, then HTTP and every rank to become idle;
3. calls `begin`, clearing statistics and enabling collection;
4. issues the manifest's `num_samples` generation requests with `do_sample=False`
   and `ignore_eos=True`;
5. waits for all requests and ranks to become idle, then snapshots and stops
   collection;
6. merges TP data on CPU (K then V), computes scale = max / 448 (with the
   tool's zero-max fallback scale of 1.0), atomically replaces report and JSON,
   and writes JSON last;
7. terminates only the service process group it created.

This is completion-based collection, not run-until-4000-KV-writes. Get KV layer
counts from the runtime; total transformer layers need not be full-attention KV
layers.

## Accept or reject the result

Accept only when all evidence is present:

- process exits 0 and all manifest `num_samples` requests succeed;
- report matches the manifest's seed, random-input setting, sample count,
  concurrency, maximum generated tokens, TP, DP, and requested main/draft
  paths (the unmodified baseline is seed 42, 100 samples, concurrency 16,
  max-new 2000, TP 4, DP 1);
- report `output_tokens`, or legacy `response_records` total, equals manifest
  `num_samples × max_new_tokens`. Report any shortfall; call it a complete
  2026-09-15 baseline only when the relevant manifest settings are unchanged;
- every TP rank has data and every layer's `counts` and `observed_token_rows`
  are positive;
- KV metadata uses the actual target + draft layer count, shape
  `[num_layers, 2 × num_head]`, `qmin=-448`, `qmax=448`, and finite positive
  scales;
- QKV output has Q metadata/shape accepted by the runtime and positive Q
  observations per layer. Native Q KV heads can differ from TP-expanded KV
  heads; allow a tool-supported divisible layout instead of forcing equality.

Validate Q stages against current model support. DSpark draft prefill need not
be full attention, so Q normally reflects decode; do not require every layer in
both prefill and decode.

On failure, preserve logs and manifest and explain the failing condition. Do
not call a partial artifact successful. You may diagnose and retry a changed
environment, but retain the data budget unless authorized otherwise; stop after
the same cause repeats rather than retrying indefinitely.

## Final response

Return JSON, report, service-log, run-directory, and manifest paths; samples,
generated-token total, rank/layer counts, actual layout, parameter deviations,
and completion status. Do not claim a new model matches historical counts,
input-token total, or input hash.
