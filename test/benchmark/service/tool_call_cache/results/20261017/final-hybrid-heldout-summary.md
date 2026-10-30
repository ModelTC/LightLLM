Final serving source, complete primary and hardware-crossover Qwen3.5-27B TP2 heldout study: all nine prescribed cases, 54 paired trials, three independent seed clusters. Both policies used identical paired work; no timed logged TileLang compilation or selected-GPU violations. Trace-shaped synthetic closed-loop load, not original production payload replay.

Paired ratios; 95% descriptive bootstrap intervals clustered by workload seed; 10000 resamples.

| Case | TTL(s) | Pairs/seeds | Wall speedup [95% CI] | p95 TTFT speedup [95% CI] | Uncached input reduction [95% CI] |
|---|---:|---:|---:|---:|---:|
| abandoned | 30 | 6/3 | 1.007x [1.006, 1.007] | 0.995x [0.985, 1.002] | +0.23% [+0.00, +0.40] |
| high-pressure-c24 | 30 | 6/3 | 1.000x [0.999, 1.001] | 1.003x [0.976, 1.022] | +0.00% [+0.00, +0.00] |
| low-pressure | 30 | 6/3 | 1.003x [1.002, 1.006] | 1.004x [0.993, 1.010] | +0.00% [+0.00, +0.00] |
| mixed25 | 30 | 6/3 | 1.021x [0.997, 1.060] | 1.001x [1.000, 1.002] | +3.01% [+0.00, +7.38] |
| ordinary-c8 | 30 | 6/3 | 0.996x [0.994, 1.000] | 0.992x [0.977, 1.003] | -0.08% [-0.24, +0.00] |
| slow-tools | 30 | 6/3 | 0.999x [0.998, 1.001] | 1.002x [1.001, 1.004] | -1.28% [-2.38, +0.00] |
| trace-c16 | 30 | 6/3 | 1.019x [1.011, 1.026] | 0.999x [0.990, 1.011] | +2.80% [+1.75, +4.04] |
| trace-c8 | 30 | 6/3 | 0.978x [0.949, 0.999] | 1.002x [0.999, 1.006] | +1.89% [-4.21, +5.79] |
| variable-output-trace | 30 | 6/3 | 1.005x [0.999, 1.016] | 0.998x [0.994, 1.003] | +1.63% [+0.00, +4.90] |

Uncached input = API prompt tokens minus cached tokens, excluding additional recomputation after pauses. Absolute timings include equal debug telemetry and the declared scaled/capped tool waits. Per-request workload descriptors are not additional independent repetitions. All audited groups and all paired trials are retained.
