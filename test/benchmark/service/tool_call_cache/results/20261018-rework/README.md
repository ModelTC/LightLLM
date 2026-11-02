# Exploratory rework validation, 2026-10-18

These are 84 complete exploratory/validation comparisons, including 9 with the repaired exact-output harness. Sources and harness definitions stay separate; all neutral and negative outcomes are included. No complete independent-corpus or GPU-crossover performance evidence is included here; this cannot establish a production benefit.

All completed comparisons below use Qwen3-4B BF16, TP=1, isolated H200 GPUs on a shared host, KV capacity 32,768 tokens, request limit 16,384 tokens, prefill batch 8,192 tokens and 32 running requests. Both policies use the same pinned existing image, attention backends, debug logging and dependency overlay. Physical KV capacity is identical; static-headroom changes admission only. No images were built. Full per-group image/model/dependency fingerprints and configuration accompany the numerical exports. Weight/tokenizer provenance remains in [the preceding study](../20261017/model-checksums.json).

The source is pinned public [CacheWise metadata](https://github.com/cachewise-project/cachewise-coding-traces/tree/181c435a090d328d00bbbee4c8eeb27d32f3abd2). Text is synthetic; actual generated tool results close the agent loop. Lengths, waits and controlled mixtures are explicitly scaled/capped. These are closed-loop workloads, not verbatim production traffic or an open-loop overload test. The v11–v15 clients reset compacted history to the same system/reference scope; identical synthetic filler can therefore be reused across compactions. The v16/v17 clients use fresh context identity before the long reference at each compaction. These two definitions are reported separately. Source snapshots differ between studies; replay namespaces differ too. Comparisons are paired within each phase, and studies are never pooled.

v11 has two validation seeds per case; v13/v14/v15r1 have one. Intervals, when present, are descriptive seed-cluster bootstrap intervals. Requests are not independent performance repetitions; there is no GPU crossover or independent-corpus performance proof here. v12 stopped during warmup on unexpected end states and has zero measured pairs. Raw requests, responses and logs remain private in the experiment store.

## Complete results

Ratios above 1 favor the candidate. A positive result against deliberately reduced static-headroom admission is not a win against ordinary disabled caching. Control columns explicitly distinguish them.

### budget-v11-validation.json.gz

| Candidate | Control | Case | Pairs / seeds | Wall ratio | Uncached input reduction (%) |
|---|---|---|---:|---:|---:|
| budget-only | historical eviction policy | mixed25-c16 | 2 / 2 | 1.0011x | 2.23 |
| budget-only | historical eviction policy | ordinary-c8 | 2 / 2 | 0.9979x | 0.00 |
| budget-only | historical eviction policy | pressure-c24 | 2 / 2 | 0.9999x | -0.11 |
| budget-only | historical eviction policy | tool-wait-c8 | 2 / 2 | 1.0072x | 2.18 |
| budget-only | historical eviction policy | trace-c16 | 2 / 2 | 1.0044x | -0.58 |
| budget-only | historical eviction policy | trace-c8 | 2 / 2 | 1.0003x | 4.03 |

[All aggregate and per-trial numeric metrics](budget-v11-validation.json.gz).

### stable-prefix-v13-validation.json.gz

| Candidate | Control | Case | Pairs / seeds | Wall ratio | Uncached input reduction (%) |
|---|---|---|---:|---:|---:|
| stable-prefix | budget-only | agent25-c16-fast | 1 / 1 | 1.0246x | 1.62 |
| stable-prefix | budget-only | agent25-c16-slow | 1 / 1 | 1.0002x | -1.48 |
| stable-prefix | budget-only | agent50-c16-fast | 1 / 1 | 1.0018x | 0.00 |
| stable-prefix | budget-only | trace-c16 | 1 / 1 | 1.0031x | -7.14 |
| stable-prefix | same source, disabled | agent25-c16-fast | 1 / 1 | 1.0193x | 1.62 |
| stable-prefix | same source, disabled | agent25-c16-slow | 1 / 1 | 1.0042x | -1.56 |
| stable-prefix | same source, disabled | agent50-c16-fast | 1 / 1 | 1.0035x | 0.00 |
| stable-prefix | same source, disabled | ordinary-c8 | 1 / 1 | 1.0005x | 0.00 |
| stable-prefix | same source, disabled | pressure-c24 | 1 / 1 | 0.9972x | 0.20 |
| stable-prefix | same source, disabled | trace-c16 | 1 / 1 | 1.0000x | -0.06 |
| stable-prefix | historical eviction policy | agent25-c16-fast | 1 / 1 | 1.0076x | 0.22 |
| stable-prefix | historical eviction policy | agent25-c16-slow | 1 / 1 | 1.0097x | 0.00 |
| stable-prefix | historical eviction policy | agent50-c16-fast | 1 / 1 | 1.0249x | -1.17 |
| stable-prefix | historical eviction policy | ordinary-c8 | 1 / 1 | 0.9978x | 0.00 |
| stable-prefix | historical eviction policy | pressure-c24 | 1 / 1 | 0.9739x | -0.00 |
| stable-prefix | historical eviction policy | trace-c16 | 1 / 1 | 0.9992x | 2.42 |

[All aggregate and per-trial numeric metrics](stable-prefix-v13-validation.json.gz).

### reserved-kv-v14-validation.json.gz

| Candidate | Control | Case | Pairs / seeds | Wall ratio | Uncached input reduction (%) |
|---|---|---|---:|---:|---:|
| reserved-KV | static-headroom control | agent25-c16-fast | 1 / 1 | 1.1141x | 11.40 |
| reserved-KV | static-headroom control | agent25-c16-slow | 1 / 1 | 1.0549x | 5.03 |
| reserved-KV | static-headroom control | agent50-c16-fast | 1 / 1 | 1.0323x | 1.93 |
| reserved-KV | static-headroom control | ordinary-c8 | 1 / 1 | 1.0102x | -0.02 |
| reserved-KV | static-headroom control | pressure-c24 | 1 / 1 | 1.0035x | -0.00 |
| reserved-KV | static-headroom control | trace-c16 | 1 / 1 | 0.9944x | 5.18 |
| reserved-KV | stable-prefix | agent25-c16-fast | 1 / 1 | 0.9672x | 9.08 |
| reserved-KV | stable-prefix | agent25-c16-slow | 1 / 1 | 0.9326x | -0.63 |
| reserved-KV | stable-prefix | agent50-c16-fast | 1 / 1 | 0.8453x | 1.16 |
| reserved-KV | stable-prefix | trace-c16 | 1 / 1 | 0.9974x | -8.67 |
| reserved-KV | same source, disabled | agent25-c16-fast | 1 / 1 | 0.9529x | 9.41 |
| reserved-KV | same source, disabled | agent25-c16-slow | 1 / 1 | 0.9379x | 3.08 |
| reserved-KV | same source, disabled | agent50-c16-fast | 1 / 1 | 0.8542x | 2.00 |
| reserved-KV | same source, disabled | ordinary-c8 | 1 / 1 | 1.0001x | -0.01 |
| reserved-KV | same source, disabled | pressure-c24 | 1 / 1 | 0.9036x | -0.53 |
| reserved-KV | same source, disabled | trace-c16 | 1 / 1 | 0.9930x | -9.36 |

[All aggregate and per-trial numeric metrics](reserved-kv-v14-validation.json.gz).

## Causal accounting

Final archived counters cover 2,072 candidate requests and 1,062 actual tool calls in each of v13 and v14, including one 24-request warmup. All 1,062 tool calls were detected in each. v13 lost 3,693,774 protected token leases through forced eviction. v14 reduced that count to zero, but budget demotions increased from 1,668,794 to 5,296,127 and consumed waiting leases decreased from 839,020 to 750,275 tokens. These are cumulative accounting events, not unique compute savings; workloads have equal prescribed cases but cross-study namespace tokenization differs. The result demonstrates that preventing forced eviction alone does not establish an end-to-end gain.

In the v14 agent25-fast comparison against disabled caching, agent uncached input fell 35.91%, yet mean agent request ratio was 0.9843x and independent ordinary request ratio was 0.8801x. Thus cache metrics alone cannot support a latency claim. The next candidate addresses returning continuations being queued behind the reservation. Its results must be reported separately.

[Complete final numeric counters](cache-counters-v13-v14.json). Earlier live diagnostic totals were partial and must not be presented as whole-study totals.

The benchmark-only static control is `0e68cc17feecbcf85e43d9a69c7ebe9471e10bab`, with retention disabled and constant 50% admission headroom. It retains the same physical cache, request and batch limits. This source is isolated on the `cache-headroom-control` branch and must not be merged as production behavior.

## Candidate with continuation priority: v15r1

These are one-seed validation results under the earlier shared-compaction identity assumption. They are not pooled with earlier sources. The service adds prefix-aware borrowing and bounded successful bypasses; all physical cache/batch/request limits remain identical.

| Control | Case | Pairs / seeds | Wall ratio | Uncached input reduction (%) |
|---|---|---:|---:|---:|
| static headroom (retention off) | agent25-c16-fast | 1 / 1 | 1.3766x | 20.76 |
| static headroom (retention off) | agent25-c16-slow | 1 / 1 | 1.2873x | 13.79 |
| static headroom (retention off) | agent50-c16-fast | 1 / 1 | 1.6116x | 34.55 |
| static headroom (retention off) | ordinary-c8 | 1 / 1 | 1.0104x | -0.02 |
| static headroom (retention off) | pressure-c24 | 1 / 1 | 1.3420x | 44.07 |
| static headroom (retention off) | trace-c16 | 1 / 1 | 1.0149x | 25.12 |
| same source, disabled | agent25-c16-fast | 1 / 1 | 1.1304x | 13.26 |
| same source, disabled | agent25-c16-slow | 1 / 1 | 1.1107x | 8.93 |
| same source, disabled | agent50-c16-fast | 1 / 1 | 1.2913x | 32.56 |
| same source, disabled | ordinary-c8 | 1 / 1 | 0.9994x | 0.00 |
| same source, disabled | pressure-c24 | 1 / 1 | 1.1274x | 49.60 |
| same source, disabled | trace-c16 | 1 / 1 | 0.9780x | 12.96 |
| reservation without continuation priority | agent25-c16-fast | 1 / 1 | 1.2344x | 8.79 |
| reservation without continuation priority | agent25-c16-slow | 1 / 1 | 1.2088x | 7.92 |
| reservation without continuation priority | agent50-c16-fast | 1 / 1 | 1.4852x | 28.68 |
| reservation without continuation priority | trace-c16 | 1 / 1 | 1.0070x | 23.96 |

[All candidate aggregate/per-trial metrics](continuation-v15r1-validation.json.gz). Ratios against the deliberately smaller static admission headroom are separate from disabled-cache-policy comparisons.

Ordinary cohort tail latency is a real tradeoff. In agent50-fast versus disabled, mean agent request ratio is 2.3037x and mean ordinary request ratio is 1.1314x, but ordinary p95 request ratio is 0.8494x and ordinary p95 TTFT ratio is 0.8024x. In agent25-slow, mean ordinary request ratio is 0.9424x. Do not claim that every latency percentile improves or that priority is free. Cohort metrics and all neutral/negative cases remain in the JSON.

Final counters now cover all 2,072 candidate requests / 1,062 actual tool calls for each of three versions. Candidate v15r1 consumed 3,388,092 token leases versus 750,275 in reservation-only v14. Budget demotions fell to 2,480,859; forced protected evictions rose to 327,736. Allowing useful continuations to run is more valuable here than reducing eviction events to zero. These remain cumulative accounting events, not unique compute savings. [Full final counters](cache-counters-v13-v15r1.json).

The completed pressure pair has 163 actual append continuations: cached tokens were 612,137 versus 9,201. Five context-reset requests had only 25 versus 26 cached tokens, so that observed gain did not come from repeated filler after compaction. [Numerical append/reset audit](pressure-append-diagnostic-v15r1.json). Independent-corpus performance under fresh identity is separate from these validation results; the old frozen clients and results remain intact.

## Fresh compaction identity: v16 validation

This uses the same priority source `5bae2ef5ea481bbf53da26007169c94737d86062` with the more conservative fresh compaction identity. All six cases were prescribed before the run. They remain separate from shared-identity results. The parent controller was intentionally aborted after these six valid pairs and before its independent-test gate; no TraceLab inference was run.

| Case | Pairs / seeds | Wall ratio | Uncached input reduction (%) |
|---|---:|---:|---:|
| abandoned-agent25-c16 | 1 / 1 | 0.9793x | 0.04 |
| agent50-c16-fast | 1 / 1 | 1.2778x | 31.00 |
| expired-agent25-c16 | 1 / 1 | 1.0165x | -0.03 |
| output-shaped-agent50-c8 | 1 / 1 | 1.1667x | 30.97 |
| pressure-c24 | 1 / 1 | 1.1474x | 46.48 |
| realwait-trace-c16 | 1 / 1 | 1.0044x | 34.19 |

[All aggregate/per-trial/cohort metrics](priority-fresh-v16-validation.json.gz). The output-shaped case uses explicitly scaled/capped source output targets and matched actual generated token counts.

The ordinary p95 TTFT ratios are 0.9050x in agent50-fast and 0.6830x in the output-shaped case despite improved means. Abandoned chains give 0.9793x wall ratio and ordinary p95 TTFT 0.8973x. These tradeoffs motivated evaluating FIFO-feasible-head-first backfill instead of unconditional promotion. These six pairs are valid diagnostic evidence, not independent heldout results, and must not be silently discarded because the parent review stopped.

## FIFO-first blocked-head backfill: v17 validation

Frozen service source: `e6d00d6cf8df7ea013a6ea6706c53d4140eb22ec`. Six conservative fresh-compaction validation pairs versus same-source Off, plus three direct pairs versus predecessor `5bae2ef5...`. Seed42 only; these are validation evidence, not independent-corpus results. All source prompts, finish states, output signatures and generated token counts match.

Against same-source Off (TTL0)

| Case | Wall ratio | Uncached input reduction (%) | Ordinary mean request ratio | Ordinary p95 TTFT ratio |
|---|---:|---:|---:|---:|
| abandoned-agent25-c16 | 0.9717x | 0.00 | 0.9547x | 0.9048x |
| agent50-c16-fast | 1.2061x | 26.36 | 1.0482x | 0.8092x |
| expired-agent25-c16 | 0.9972x | 0.43 | 0.9080x | 0.8666x |
| output-shaped-agent50-c8 | 1.1686x | 31.00 | 1.1198x | 0.6747x |
| pressure-c24 | 1.2044x | 48.42 | n/a | n/a |
| realwait-trace-c16 | 1.0070x | 29.82 | n/a | n/a |

[All aggregate/per-trial/cohort metrics](fifo-v17-validation.json.gz).

Against predecessor unconditional priority (TTL3)

| Case | Wall ratio | Uncached input reduction (%) | Ordinary mean request ratio | Ordinary p95 TTFT ratio |
|---|---:|---:|---:|---:|
| agent50-c16-fast | 0.9957x | -2.19 | 0.9982x | 1.0675x |
| output-shaped-agent50-c8 | 1.0120x | -0.06 | 1.0066x | 1.0092x |
| pressure-c24 | 0.9106x | -10.67 | n/a | n/a |

[All aggregate/per-trial/cohort metrics](fifo-v17-priority.json.gz).

[Cohort tradeoff chart](fifo-v17-validation-cohorts/Qwen3-4B.png) ([SVG](fifo-v17-validation-cohorts/Qwen3-4B.svg)) shows every validation case, including unavailable cohort metrics and negative ratios.

The faster fast-agent/pressure results against Off coexist with ordinary tail regressions. FIFO-first is a bounded fairness choice: against unconditional priority it improves fast-agent ordinary p95 TTFT by 1.0675x, while agent mean request ratio is 0.8396x and pressure wall ratio 0.911x. It does not eliminate the cost of retaining idle KV. Slow tools and abandoned chains provide little or no wall benefit. The feature remains disabled by default.

[Policy freeze before independent inference](fifo-v17-policy-freeze.json) records these limitations, the exact service/replay revisions and validation report hashes. Independent TraceLab assessment, hardware/order crossover and TP2 hybrid validation are in progress; do not interpret these nine pairs as production evidence.

## Independent metadata: observed return timing

[TraceLab v0.0.1](https://github.com/uw-syfi/TraceLab#-the-dataset) supplies sanitized real coding-agent metadata, not original prompts. On the pinned normalized longest valid segment per session (348,158 rounds), **all original finish reasons are missing**. Tool presence is a separate advisory label.

Among observed valid input-readiness gaps, tool rounds have a median of **0.224s**, versus **130.593s** without observed tools; **76.84%** versus **4.62%** are within three seconds. These denominators are 311,225 and 31,602 observed gaps respectively. Next first-response-chunk gaps instead have medians 5.958s and 149.029s: these include serving latency and are not request-arrival times.

[Complete numerical timing audit](tracelab-observed-returns-v17.json) includes all gap distributions, missing followups and conditional fractions. Final/segment-boundary observations are censored; lack of a followup is not abandonment, and no HTTP-arrival, token-identity or causal performance inference is justified. This supports the return-timing hypothesis, while end-to-end gains remain a separate question. The three-second policy had already been frozen before independent model inference; this analysis does not change it.

## Excluded first independent attempt: strict output parity

The v17 independent phase stopped in agent25-fast seed42: one ordinary request with the same prompt and five cached tokens in both policies produced punctuation variation and two versus three completion tokens. Both ended normally. The six preceding matched pairs and the failed pair remain private in the original phase; **the whole incomplete phase is excluded from final performance reporting**. No crossover or hybrid service was started, and all owned services were cleaned up. [Numerical exclusion record](excluded-v17-independent-parity.json).

The benchmark repair adds optional `--ordinary-output json-string` using an exact-string JSON schema for both short and shaped ordinary replies. Existing free-text mode remains the default for historical reproducibility. Strict prompt, output-signature and generated-token equality remains required. The complete validation and independent plans must restart under the new frozen harness; no failed pair is dropped or normalized into a success, and the cache source and policy remain unchanged.

## Excluded JSON-string validation: output budget

The first constrained-output validation stopped on a source-shaped ordinary reply in the disabled control: the inherited estimate was too short for a complete JSON string. No independent requests ran. The entire incomplete validation phase is excluded; all owned services were cleaned up. [Exclusion record](excluded-v18-output-budget.json).

The repair sizes the completion allowance from the actual tokenizer and serialized literal rather than the source-shaped estimate alone. [Actual-tokenizer audit](tokenizer-budget-v19.json) checks targets 0–512 for both Qwen3-4B and Qwen3.5-27B: 275 of 513 target lengths exceeded the old allowance; all new allowances fit the complete literal and EOS while staying below the request limits. These are untimed correctness checks, not performance evidence. Actual generated counts remain measured and must match between policies.

## Complete repaired-harness validation: v20

Same frozen cache source `e6d00d6cf8df7ea013a6ea6706c53d4140eb22ec`, same 3s lease / 0.5 budget. Benchmark `b491a8807239a3cfdc091613b79d23cd403b44cd` uses exact ordinary JSON strings with actual-tokenizer/literal/EOS budgets. Every service received 32 long ordinary preflight requests before all paired warmups and timing. All 9 prespecified pairs pass exact prompts, signatures, generated-token counts and GPU isolation. Seed42 only; not independent results. Never pool with earlier free-text or incomplete phases.

Against same-source Off (TTL0)

| Case | Wall ratio | Uncached input reduction (%) | Agent mean request ratio | Ordinary mean request ratio | Ordinary p95 TTFT ratio |
|---|---:|---:|---:|---:|---:|
| abandoned-agent25-c16 | 0.9790x | -0.04 | 1.0938x | 0.9502x | 0.9673x |
| agent50-c16-fast | 1.1907x | 24.68 | 1.8807x | 1.0871x | 0.8221x |
| expired-agent25-c16 | 0.9983x | -0.02 | 0.9185x | 0.9149x | 0.9066x |
| output-shaped-agent50-c8 | 1.1484x | 29.89 | 1.7047x | 1.1196x | 0.7698x |
| pressure-c24 | 1.2291x | 49.28 | n/a | n/a | n/a |
| realwait-trace-c16 | 1.0492x | 28.20 | n/a | n/a | n/a |

[All aggregate/per-trial/cohort metrics](fifo-v20-validation.json.gz).

Against predecessor unconditional priority (TTL3)

| Case | Wall ratio | Uncached input reduction (%) | Agent mean request ratio | Ordinary mean request ratio | Ordinary p95 TTFT ratio |
|---|---:|---:|---:|---:|---:|
| agent50-c16-fast | 0.9146x | -12.60 | 0.8601x | 0.9173x | 0.9399x |
| output-shaped-agent50-c8 | 1.0150x | -1.27 | 1.0238x | 1.0198x | 1.0048x |
| pressure-c24 | 1.0136x | -3.46 | n/a | n/a | n/a |

[All aggregate/per-trial/cohort metrics](fifo-v20-priority.json.gz).

[Repaired validation cohort chart](fifo-v20-validation-cohorts/Qwen3-4B.png) ([SVG](fifo-v20-validation-cohorts/Qwen3-4B.svg)) shows wall time, agent mean and ordinary p95 together.

Faster replay and agent means coexist with ordinary tail costs. Expired/abandoned cases have little or no cache benefit. This remains opt-in; no universal latency or production-speed claim is justified. The prior source/policy was fixed before the first independent inference, and only benchmark correctness was repaired after the excluded attempts. All independent cases, seeds, caps and GPU/order assignments remain prescribed.

[Repaired-harness validation freeze](fifo-v20-policy-freeze.json). Independent results will be reported separately after the complete phases and crossovers pass.

## Fixed independent-transfer protocol

The [normal/default-off protocol](transfer-protocol-v20.json), [TP2 hybrid protocol](hybrid-transfer-protocol-v20.json) and [frozen benchmark file hashes](harness-transfer-v20.json) retain the prescribed cases, seeds, caps, GPU/order assignments and source/dataset revisions. Planning-status text records the state when prepared; actual completion is determined only by full audited phases. The ordinary-output correctness repairs and excluded attempts are explicit. Cache source and normal 3s / 0.5 policy remain fixed since before the original independent inference; no independent performance tunes them.

There are 54 normal independent pairs, 12 default-off/upstream pairs and 24 hybrid TP2 pairs planned. Incomplete phases and untimed correctness/preflight requests never enter those counts. Hybrid retains the explicitly prespecified validation-only rule for considering a longer window before any hybrid independent run; any chosen configuration must match across GPU roles.
