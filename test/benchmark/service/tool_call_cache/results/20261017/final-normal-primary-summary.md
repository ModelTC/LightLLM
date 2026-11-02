Final-source Qwen3-4B primary heldout only; candidate GPU0, control GPU1, TTL3s/fraction0.5. All50 pairs across10 prespecified cases and5 workload seeds. Fixed physical assignment; hardware crossover and default-off audit remain pending. Source dcd3551c53a13c5613d8a8b644870de00dd57820; equal debug instrumentation. No stable production speedup established.

Paired ratios; 95% descriptive bootstrap intervals clustered by workload seed; 10000 resamples.

| Case | TTL(s) | Pairs/seeds | Wall speedup [95% CI] | p95 TTFT speedup [95% CI] | Uncached input reduction [95% CI] |
|---|---:|---:|---:|---:|---:|
| abandoned | 3 | 5/5 | 1.001x [0.999, 1.003] | 1.009x [0.981, 1.040] | -0.85% [-4.17, +2.91] |
| high-pressure-c24 | 3 | 5/5 | 1.000x [0.994, 1.006] | 1.012x [0.984, 1.040] | -0.64% [-4.16, +3.48] |
| low-pressure | 3 | 5/5 | 0.997x [0.993, 1.001] | 0.961x [0.866, 1.041] | -0.16% [-0.47, +0.00] |
| mixed25 | 3 | 5/5 | 1.001x [1.000, 1.003] | 1.020x [0.829, 1.222] | +0.57% [-0.03, +1.74] |
| ordinary-c8 | 3 | 5/5 | 0.998x [0.994, 1.001] | 0.927x [0.739, 1.176] | -10.72% [-39.06, +6.72] |
| slow-tools | 3 | 5/5 | 1.000x [0.998, 1.001] | 1.060x [1.007, 1.111] | -0.01% [-0.04, +0.00] |
| trace-c16 | 3 | 5/5 | 1.000x [0.992, 1.008] | 1.015x [0.971, 1.060] | +4.31% [+0.06, +8.63] |
| trace-c8 | 3 | 5/5 | 1.003x [1.000, 1.007] | 0.904x [0.754, 1.064] | +0.02% [+0.00, +0.06] |
| variable-output-ordinary | 3 | 5/5 | 1.000x [0.998, 1.002] | 1.035x [0.953, 1.167] | +2.18% [+0.00, +6.53] |
| variable-output-trace | 3 | 5/5 | 0.999x [0.998, 1.000] | 1.029x [0.934, 1.159] | -0.02% [-0.06, +0.00] |

Uncached input = API prompt tokens minus cached tokens, excluding additional recomputation after pauses. Absolute timings include equal debug telemetry and the declared scaled/capped tool waits. Per-request workload descriptors are not additional independent repetitions. All audited groups and all paired trials are retained.
