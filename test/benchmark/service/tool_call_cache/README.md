# Tool-call-aware KV retention experiments

This directory provides complementary kinds of evidence:

* `trace_data.py`: observations from pinned, public, sanitized CacheWise sessions.
  Multiple assistant chunks are grouped by request ID. Missing original finish
  reasons remain missing. Tool-content labels are reported separately. Final
  observations are censored; return probabilities are conditional on observing
  another request. Input readiness and the next response's first chunk are
  different timing measurements, neither an exact HTTP arrival timestamp.
* `tracelab_data.py`: independent [TraceLab v0.0.1 metadata](https://github.com/uw-syfi/TraceLab/releases/tag/v0.0.1),
  pinned by SHA-256. Reject missing usage and generation timing; retain one longest
  contiguous, nonoverlapping segment per original session, choosing the first on
  ties. Do not bridge invalid rounds or index gaps. Normalize identities to hashes
  and tool presence to a fixed label. Missing original finish reasons and input
  readiness remain missing; usage-report timestamps are not generation events.
  These sanitized metadata still require synthetic replay content. Preserve CC
  BY 4.0 attribution. Reserve an independent corpus before performance tuning.
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
  and server TTFT/cache metrics correlated by request ID. Dataset checksums and
  both policy settings separate report groups. Older protocols lack the explicit
  dataset checksum; do not pool them with new independent-corpus evaluations. Comparison stops on
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
* `plot_results.py --cohorts`: displays wall time, agent mean request latency and
  ordinary one-shot p95 TTFT together from the audited numerical export. Ratios
  below one expose costs even when total wall time improves. Missing cohorts
  remain visibly unavailable.
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

`--mix agent25` and `--mix agent50` assign respectively one quarter and one
half of the seeded client streams to multi-round agents. Agents call tools until
the final answer; the remaining streams issue independent ordinary requests,
each with a fresh namespace before its long reference context. This differs
from `mixed25`, which randomly changes the finish state within repeatedly reused
conversations and deliberately makes ordinary requests hot too. Both are
controlled synthetic mixtures, not observations of production request classes.
The original trace input sizes/waits and any overrides remain recorded.
Experimental isolation labels are hashed before entering the system prompt;
policy names are never model instructions. The system explicitly treats earlier
user instructions as completed and requires real calls instead of textual tool
imitations. Unexpected end states remain failed trials, including warmup.

`--compaction-identity fresh` is the default for new replays: source context drops
and prompt-cap resets replace the system scope before the long synthetic
reference. Redacted metadata cannot establish a shared long prefix across these
resets. `--compaction-identity shared` explicitly reproduces the earlier
assumption, which can reuse identical filler after resetting history. Preserve
this sensitivity and the harness hash in each report; do not pool both modes.
The 2026-10-17 study and v11/v13/v14/v15r1 exploratory snapshots predate this
option and use shared compaction identity. Their archived clients remain intact.

Mixed-case `session_seconds` describes complete client streams, including streams
of independent ordinary requests; it is not an agent-only completion metric.
Use `agent_session_seconds` for completed agents. Numerical cohort metrics report
agent and one-shot cache work, request latency and server TTFT separately, so a
cache preference cannot hide latency transferred to unrelated ordinary requests.

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

Ordinary free-text responses can differ by punctuation across batches despite
temperature zero. For strict equal-work studies, pass `--ordinary-output
json-string` in every case: short and source-shaped ordinary replies then use
an exact-string JSON schema, with a completion allowance based on the actual
tokenized literal and conservative byte/EOS margin. Source-shaped targets are
estimates; actual generated tokens remain measured. This changes the synthetic workload and requires
a new harness fingerprint and complete rerun. Keep signature and generated-token
equality checks; do not normalize a failed pair into a performance win.

## Feature configuration

`--tool_call_cache_retention_seconds 30 --tool_call_cache_max_fraction 0.5`
enables temporary eviction preferences after normally stopped generations that
contain the configured parser's tool-start marker. Only tool-enabled chat
requests supply that marker. `length`, abort/error and paused requests receive
no new preference. The default duration of zero preserves ordinary eviction.
When the template supports a history-only render, the lease ends before its
short generation-only assistant header. That header and generated tool JSON
can be rewritten when clients serialize the next tool-result request. The
inference worker protects the stable input history rather than waiting for an
unmatchable generated tail. Unsupported custom renders retain the advisory
fallback; token-exact cache lookup remains authoritative.

A continuation that acquires a real KV reference consumes its waiting lease
immediately; the pinned request no longer competes for the waiting budget.
Budget pressure trims older continuations incrementally at page boundaries,
retaining their useful prefix instead of discarding an entire shared subtree.
Hybrid attention trims only to checkpoints with valid recurrent state.
Numeric debug counters distinguish consumed, budget-demoted, expired, completed,
and evicted lease tokens. These are accounting events, not unique prefill savings.

Normal-service admission now accounts for protected KV that is actually idle.
Without this, treating the entire cache as free admits enough unrelated active
contexts to force out waiting tool prefixes despite their eviction preference.
Shared ancestors count once, and referenced KV does not also consume this idle
reservation. An initial feasible request or an explicitly high-priority request
can borrow the reservation, so a large request need not wait for lease expiry.
A bounded shared table publishes at most 64 prefix fingerprints and deadlines
for prefixes of at least 256 tokens, excluding tiny shared scaffolding,
at request-release batches. A single-host router can recognize a waiting
continuation and admit it against the original full-capacity bound. A feasible
FIFO head always runs first. Only when that head is blocked does the queue scan
for cached continuations that fit; at most eight successful admissions may bypass
the same blocked head. Failed probes preserve FIFO and the capacity estimate. Explicit caller priority
remains first. Fingerprints are scheduling hints only: actual KV reuse still
requires the radix cache's exact match. Stale/missing hints fall back to ordinary
admission. Local hint ordering is disabled for multinode TP and PD roles.
The backend can still reclaim every protected token; no reference is added.
This reservation may reduce active concurrency while tools are pending, so the
cohort latency and abandoned/slow-tool controls remain necessary. PD admission
is unchanged because separate prefill has no tool-completion feedback.

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

## Previous implementation study (2026-10-17–18)

The serving source measured in this historical study is `dcd3551c53a13c5613d8a8b644870de00dd57820`;
these results do not measure the later waiting-budget and stable-prefix reimplementation.
the unchanged upstream control is `22888b4237aa2c895fc24176e08089cf14bc6c89`.
All 238 required paired comparisons completed: 36 validation, 154 held-out
including hardware crossover, 20 default-disabled versus upstream, and 28
controlled residency checks. Earlier-source supplemental results and excluded
thermal/compiler-contaminated attempts remain separate.

These scaled, closed-loop workloads do not establish general acceleration.
Qwen3.5-27B trace-c16 wall speedup was 1.019x [1.011, 1.026], while trace-c8
was 0.978x [0.949, 0.999]. Qwen3-4B results were near 1x, including a small
high-pressure regression. The default-disabled trace-c16 p95 TTFT ratio was
0.953x [0.905, 0.994], so this study does not establish zero overhead.
Intervals are descriptive seed-clustered estimates with only three or five
independent workload seeds; crossover repeats share a seed cluster.
Retention remains disabled by default.

| Evidence | Pairs | Complete results |
|---|---:|---|
| Qwen3-4B held-out crossover | 100 | [Table](results/20261017/final-normal-heldout-summary.md), [numeric trials](results/20261017/final-normal-heldout-summary.json.gz) |
| Qwen3.5-27B held-out crossover | 54 | [Table](results/20261017/final-hybrid-heldout-summary.md), [numeric trials](results/20261017/final-hybrid-heldout-summary.json.gz) |
| Default-disabled versus upstream | 20 | [Table](results/20261017/final-default-off-summary.md), [numeric trials](results/20261017/final-default-off-summary.json.gz) |
| Ordinary-attention controlled residency | 25 | [Table](results/20261017/final-normal-pressure-summary.md), [numeric trials](results/20261017/final-normal-pressure-summary.json) |
| Hybrid actual tool-result continuation | 3 | [Table](results/20261017/final-hybrid-pressure-summary.md), [numeric trials](results/20261017/final-hybrid-pressure-summary.json) |

The controlled hybrid check retained a valid 32,768-token checkpoint versus
zero cached tokens in the disabled control after capacity churn. Its 120-second
window and 25% budget are a separate mechanism diagnostic, not the held-out
30-second/50% setting or a production speedup claim. Ordinary controlled checks
repeat the initial prompt to isolate residency; the hybrid check sends the real
generated tool result and a subsequent tool instruction.

[Validation/selection protocol](results/20261017/final-study-protocol.json),
[actual backend parity](results/20261017/final-backend-parity.json), and
[post-run integrity/cleanup audit](results/20261017/final-integrity-audit.json)
record the full scope. Source snapshots, all model files and the 353-file
runtime overlay passed byte-level checks; all 12 phases passed selected-GPU
monitoring and final-sample checks. The earlier ordinary controlled probe lacks
an independently timestamped initial boundary, which is explicitly identified
in the audit. All own model containers and the last four GPU processes exited.

The host was shared; selected GPUs were isolated, while CPU and I/O were not
exclusive. Namespace isolation reduces cross-session sharing. Source text,
output targets and waits were synthesized/scaled/capped rather than replayed
verbatim from production. Known timed TileLang compilation caused old hybrid
attempts to be quarantined and repeated; the guard covers logged TileLang,
not every possible runtime compiler. Intermittent pinned-memory startup
failures were preserved and recovered at unchanged precision/capacity/source;
they contributed no measured trials. See [setup/exclusion records](results/20261017/setup-failures.json).
