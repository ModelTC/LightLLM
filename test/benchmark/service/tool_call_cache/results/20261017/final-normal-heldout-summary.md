Final-source Qwen3-4B heldout with hardware crossover; all100 pairs across10 prespecified cases and5 independent workload seeds, TTL3s/fraction0.5. Primary candidate GPU0/control GPU1; crossover candidate GPU1/control GPU0. Same-seed assignments clustered together. Source dcd3551c53a13c5613d8a8b644870de00dd57820, equal debug telemetry. No stable general end-to-end acceleration established.

Paired ratios; 95% descriptive bootstrap intervals clustered by workload seed; 10000 resamples.

| Case | TTL(s) | Pairs/seeds | Wall speedup [95% CI] | p95 TTFT speedup [95% CI] | Uncached input reduction [95% CI] |
|---|---:|---:|---:|---:|---:|
| abandoned | 3 | 10/5 | 1.001x [0.999, 1.003] | 0.991x [0.933, 1.053] | -0.71% [-2.51, +1.05] |
| high-pressure-c24 | 3 | 10/5 | 0.996x [0.993, 0.999] | 1.004x [0.997, 1.016] | -0.22% [-2.78, +3.92] |
| low-pressure | 3 | 10/5 | 0.999x [0.996, 1.001] | 0.992x [0.944, 1.030] | +0.41% [-0.47, +1.70] |
| mixed25 | 3 | 10/5 | 1.001x [1.000, 1.002] | 1.061x [0.917, 1.212] | +0.90% [-2.37, +4.11] |
| ordinary-c8 | 3 | 10/5 | 0.998x [0.996, 1.000] | 0.974x [0.819, 1.158] | -4.74% [-18.92, +4.00] |
| slow-tools | 3 | 10/5 | 0.998x [0.993, 1.001] | 1.054x [1.010, 1.086] | -0.01% [-0.04, +0.00] |
| trace-c16 | 3 | 10/5 | 0.998x [0.994, 1.003] | 0.988x [0.919, 1.046] | +0.80% [-4.01, +5.93] |
| trace-c8 | 3 | 10/5 | 1.003x [1.000, 1.006] | 0.952x [0.810, 1.092] | +0.02% [+0.00, +0.03] |
| variable-output-ordinary | 3 | 10/5 | 1.001x [1.000, 1.002] | 1.032x [1.005, 1.074] | +0.86% [-0.67, +3.27] |
| variable-output-trace | 3 | 10/5 | 1.001x [1.000, 1.002] | 1.020x [0.982, 1.074] | -0.01% [-0.02, +0.00] |

Uncached input = API prompt tokens minus cached tokens, excluding additional recomputation after pauses. Absolute timings include equal debug telemetry and the declared scaled/capped tool waits. Per-request workload descriptors are not additional independent repetitions. All audited groups and all paired trials are retained.
