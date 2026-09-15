# Calibration tools

`calibrate_fp8kv.py` starts a short-lived loopback-only **text** calibration
service, collects target and optional DSpark draft K/V maxima, writes calibration
JSON plus an adjacent report, then terminates only the process group it created.
It never operates an existing server. The internal export switch is not a public
server option.

## Reproducible random-token input

Set shell variables on their own lines so argument expansion is unambiguous:

```bash
MODEL=/path/to/model
DRAFT=/path/to/dspark-draft
CHAT=/path/to/chat-template
python tools/calibrate_fp8kv.py \
  --num_samples 128 --max_input_tokens 1024 --max_new_tokens 256 \
  --concurrency 4 --seed 42 --output kv_cache_calib_per_head_with_draft.json \
  -- --model_dir "$MODEL" --chat_template "$CHAT" --mtp_mode dspark \
  --mtp_draft_model_dir "$DRAFT" --mtp_step 3 --llm_prefill_att_backend fa3 --llm_decode_att_backend fa3 \
  --port 28080 --nccl_port 39080
```

Random input consists of deterministic non-special IDs inside the configured
model vocabulary. It makes calibration input repeatable; it does not claim to
match a production prompt distribution. The report records that input kind,
seed, samples, token count, rank-local layer observations, and service settings.
The defaults are 128 samples, 1024 input tokens, 256 generated tokens, and
concurrency 4. Generation uses `ignore_eos=True`, so each request runs to the
configured output limit. Pass `--overwrite` to replace an existing JSON and its
adjacent report. Each file is atomically replaced; the JSON is published last.

## JSONL input

Each JSONL record must have `prompt` or `messages`. `messages` is passed through
the selected tokenizer/chat template once before encoding. Rows are never repeated
when the file has fewer usable records than `--num_samples`.

```json
{"prompt":"Calibrate this model."}
{"messages":[{"role":"user","content":"Calibrate this model."}]}
```

```bash
python tools/calibrate_fp8kv.py --dataset prompts.jsonl --num_samples 128 \
  -- --model_dir "$MODEL" --chat_template "$CHAT" --llm_kv_type fp8kv_sph \
  --llm_prefill_att_backend fa3 --llm_decode_att_backend fa3 --port 28080 --nccl_port 39080
```

FA3 selects per-head output and accepts an explicit `fp8kv_sph` intent;
FlashInfer selects per-tensor output and accepts `fp8kv_spt`. The isolated service
uses normal unquantized KV storage while collecting. `auto` is rejected to avoid
silently exporting the wrong granularity. The first release supports a single
node, `dp=1`, and normal mode. JSON puts target layers before draft layers;
target-only jobs are supported. Vision and audio are disabled for this text-only
workflow.
