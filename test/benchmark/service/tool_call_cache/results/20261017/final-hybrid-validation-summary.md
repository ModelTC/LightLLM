Final-source Qwen3.5-27B TP2 validation only: all18 pairs,3 prespecified cases at TTL3s and30s,3 independent workload seeds, fraction0.5/cache131072. Fixed candidate GPU0/1 versus control GPU4/5. Both per-pair and independent archived-log guards verified zero measured TileLang compilation. Source dcd3551c53a13c5613d8a8b644870de00dd57820, equal debug telemetry. Validation chooses a window before heldout; not heldout or general production speedup evidence.

Paired ratios; 95% descriptive bootstrap intervals clustered by workload seed; 10000 resamples.

| Case | TTL(s) | Pairs/seeds | Wall speedup [95% CI] | p95 TTFT speedup [95% CI] | Uncached input reduction [95% CI] |
|---|---:|---:|---:|---:|---:|
| ordinary-c8 | 3 | 3/3 | 0.996x [0.931, 1.073] | 0.945x [0.839, 1.010] | -4.87% [-18.04, +5.01] |
| trace-c16 | 3 | 3/3 | 0.998x [0.991, 1.006] | 1.012x [1.007, 1.016] | -1.34% [-2.58, -0.35] |
| trace-c8 | 3 | 3/3 | 0.980x [0.896, 1.050] | 1.034x [1.002, 1.100] | -5.58% [-17.40, +0.65] |
| ordinary-c8 | 30 | 3/3 | 0.999x [0.995, 1.002] | 1.004x [1.000, 1.010] | -0.57% [-1.57, +0.00] |
| trace-c16 | 30 | 3/3 | 1.034x [1.004, 1.096] | 0.989x [0.970, 1.002] | +4.02% [-0.34, +9.69] |
| trace-c8 | 30 | 3/3 | 0.998x [0.974, 1.019] | 1.014x [0.985, 1.054] | +4.31% [+0.00, +12.09] |

Uncached input = API prompt tokens minus cached tokens, excluding additional recomputation after pauses. Absolute timings include equal debug telemetry and the declared scaled/capped tool waits. Per-request workload descriptors are not additional independent repetitions. All audited groups and all paired trials are retained.
