# Tool-call-aware KV retention experiments

This directory provides complementary kinds of evidence:

* `trace_data.py`: observations from pinned, public, sanitized CacheWise sessions.
  Multiple assistant chunks are grouped by request ID. Missing original finish
  reasons remain missing. Tool-content labels are reported separately. Final
  observations are censored; return probabilities are conditional on observing
  another request. Input readiness and the next response's first chunk are
  different timing measurements, neither an exact HTTP arrival timestamp.
* `smoke.py`: actual OpenAI tool generation and a live tool-result continuation.
  Tool definitions remain identical across turns to preserve the prompt prefix.
* `pressure_probe.py`: controlled residency diagnostic with real tool generation
  and independent, one-token completion requests that fill the cache. Repeating
  the original prompt isolates eviction from chat-history reserialization. This
  is not a production throughput result.
  `--return-mode continuation` instead sends the actual generated tool result
  and another tool instruction after the same cache pressure, exercising the
  agent loop. Opaque call IDs are normalized consistently across paired requests.
  Size numeric filler with the actual model tokenizer: a four-digit word can
  occupy five tokens including its separator. Check the resulting prompt against
  both the retention budget and the request-length limit before running an
  oversized-prefix probe. Actual server usage remains the final size check.
* `replay.py`: closed-loop, trace-shaped load against a real inference server.
  The released text is redacted, so contexts are synthetic. Actual model outputs
  feed forward into subsequent requests. Original context sizes and waits are
  scaled explicitly; wait caps and compactions are counted. Prompt-token hashes
  permit checking that paired runs received identical token sequences. The
  default uses short echo/OK outputs and does not preserve original output
  lengths. `--output-scale 0.5 --max-output-target 256` adds explicitly capped,
  source-shaped output targets; the model's actual lengths are measured and
  paired, not assumed to match those targets. Echo results feed the actual
  generated arguments back into the conversation.
* `run_paired.py`: sequential paired trials with alternating order, archived
  service manifests, an immutable client snapshot, strict prompt/output checks,
  and server TTFT/cache metrics correlated by request ID. Comparison stops on
  errors or mismatches; earlier and failed results remain available. Continuous
  GPU process/load observations reject foreign processes on either experiment
  GPU while retaining visibility into activity elsewhere on the shared host.
  Temperature, SM/memory clocks and thermal-event counters are recorded too;
  active thermal slowdown or an increase in those counters invalidates the
  comparison, including brief throttling between monitoring samples.
* `reset_cache.py`: untimed, exp-recorded isolation before each trial. Wait for
  prior preferences to expire, then fill more than KV capacity with independent
  ordinary completions. This removes prior trial contexts without enabling RL
  mode (which changes token-rank inference work). Reset requests do not enter
  replay latency/throughput metrics. `--no-reset` is for warmup/exploration only.
* `report.py`: independently verifies raw pairs and generated token counts before
  computing seed-clustered uncertainty. `--public-output FILE` exports an
  explicit whitelist of aggregate/per-trial metrics and configuration; request
  contents, responses and raw server logs stay in the experiment store.
  Current upstream emits request-correlated TTFT/cache logs at debug level.
  Use `launch_server.py --log-level debug` consistently on both policies and the
  unchanged baseline when collecting those metrics. Keep the logging setting in
  the result configuration; absolute latency includes its instrumentation cost.
* `model_provenance.py`: hashes the actual weights and tokenizer files in a
  separate untimed interval, archiving the script and results under `exp`.
* `workload_stats.py`: exports numerical input/output-length and wait
  distributions, prompt/output target caps, and recorded context resets from
  completed paired phases. It verifies identical paired workload descriptors
  and omits request IDs, contents, output signatures and private configuration.
  These per-request distributions describe the synthetic workload; they do not
  provide additional independent performance repetitions.
* `plot_results.py`: renders every aggregate group from a numerical public
  report as standalone SVG/PNG figures, retaining seed-clustered descriptive
  intervals. It does not pool groups or recompute estimates. Uncached input
  means API prompt tokens minus cached tokens; it does not count recomputation
  caused by inference pauses. Use an explicit title to distinguish supplemental
  prior-source data from final-source comparisons.

Replay negative controls include `--mix ordinary`, `--mix mixed25`,
`--abandon-fraction 0.5` (selected sessions stop after their first tool call),
and `--min-tool-wait 4 --max-wait 6` (tool waits longer than a short retention
window). Abandoned chains are reported separately and excluded from completed
session throughput and latency. The original trace labels and wait scaling are
still recorded; these overrides are controlled workloads, not original traffic.
Ordinary requests may finish with `stop` or `length`; both receive ordinary
cache treatment. Their actual finish reasons are counted separately, and paired
trials must match both the finish reason and generated token count. An unexpected
tool call in an ordinary workload remains an error.

All benchmark/evaluation commands must be wrapped in `exp -m "purpose" ...`.
Save the exact service source revision/checksum, image digest, command line,
hardware, driver, model configuration, dataset revision and dataset checksums.
Raw downloaded traces and generated results belong outside the source tree.

## Comparison protocol

1. Compare the same source with retention disabled and enabled. Separately check
   the unchanged base revision for implementation overhead and regressions.
2. Keep model, precision, attention backend, cache-token capacity, batch limits,
   CUDA graph settings, load, client code and warmup identical. Exclude compilation
   and warmup from measured results. Do not run timed trials while another server
   is starting or warming up on the host.
   For hybrid FlashQLA, persist TileLang kernels with `--tilelang-cache` on
   `launch_server.py` and use `--reject-timed-tilelang-compilation` on
   `run_paired.py`. Representative warmup alone can miss kernel variants selected
   by batch occupancy or context parallelism. The guard checks both policies'
   actual UTC replay intervals against logged compiler begin/end events and
   rejects the phase if they overlap. `compile_guard.py` independently rechecks
   archived logs, including wrapper intervals for controlled continuations.
   Its guarantee covers logged TileLang compilation, not every possible runtime
   compiler. Keep rejected attempts visible and warm missing variants outside
   measurement before repeating both policies.
3. Select sessions/windows with a fixed seed. Tune settings on the validation
   partition only; report held-out test partitions. Alternate trial order and
   exchange policy/GPU assignments to control hardware and temporal differences.
4. Include low pressure, cache pressure, overload, ordinary-only traffic,
   tool-heavy traffic, expired/abandoned continuations and oversized prefixes.
   Check TP=1 and TP>1, ordinary and hybrid attention. Record errors and unexpected
   finish states; failed runs remain visible and are not silently filtered.
   The provided replay uses closed-loop agent chains: its high-concurrency case
   tests memory pressure and queueing, not open-loop offered-load overload.
5. Report cache-hit and newly prefilled tokens, client request latency, session
   completion time and throughput. Correlate server TTFT logs by `X-Request-Id`.
   Check prompt hashes, end states and output signatures between policies before
   interpreting performance. Report all repetitions and uncertainty, including
   neutral or negative outcomes. Do not extrapolate controlled residency gains
   into production throughput gains.

## Feature configuration

`--tool_call_cache_retention_seconds 30 --tool_call_cache_max_fraction 0.5`
enables temporary eviction preferences after normally stopped generations that
contain the configured parser's tool-start marker. Only tool-enabled chat
requests supply that marker. `length`, abort/error and paused requests receive
no new preference. The default duration of zero preserves ordinary eviction.
An ordinary stop or length-limited continuation also revokes stale preferences
along its completed branch, while preserving ancestors shared by other protected
branches. Abort/error and pauses do not revoke other requests' preferences.

The token detector is advisory, independent of API parsing: quoted or malformed
tool markers may receive a preference without changing the response. Prefixes
larger than the budget retain a page-aligned partial prefix; hybrid attention
uses the largest existing resumable checkpoint within budget. Preferences
include shared ancestors, count shared KV only once, expire without refreshing
recency, and never acquire references or prevent pressure-driven reclamation.
Hybrid recurrent-state eviction follows the same preference but still reclaims
unusable leaves first. TP ranks consume a common clock through the existing
scheduling broadcast, so expiry cannot diverge due to process-local clocks.

The hint originates from tool-enabled chat requests and is applied in the
inference worker that finishes generation. This does not send decode completion
feedback to a separate prefill service in PD deployments. A PD prefill cache
therefore cannot be assumed to receive the same benefit. Native completion
requests without parser marker metadata follow ordinary eviction. Multiple
requests that share a prefix can retain that prefix until their leases expire;
an ordinary response revokes only its completed branch, not other branches.

Example trace collection and analysis (replace `TRACE_SHA` with a full SHA):

```bash
python test/benchmark/service/tool_call_cache/trace_data.py download \
  --revision TRACE_SHA --directory /tmp/tool-call-cache/data
exp -m "Analyze pinned public coding trace continuation timing" \
  python test/benchmark/service/tool_call_cache/trace_data.py analyze \
  --directory /tmp/tool-call-cache/data --output /tmp/tool-call-cache/analysis.json
```

Example replay; save the service manifest alongside the result:

```bash
exp -m "Held-out trace-shaped replay; attach exact service/model/hardware configuration" \
  python test/benchmark/service/tool_call_cache/replay.py \
  --url http://127.0.0.1:8000 --model-dir /models/Qwen3-4B \
  --rounds /tmp/tool-call-cache/analysis.rounds.jsonl \
  --split test --sessions 8 --turns 8 --seed 42 \
  --output /tmp/tool-call-cache/replay.json
```

Dataset: [CacheWise coding traces](https://github.com/cachewise-project/cachewise-coding-traces),
[paper](https://arxiv.org/abs/2606.16824). Preserve the authors' attribution and
the source revision when publishing derived results.
