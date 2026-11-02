Final-source normal-attention validation only; 3 independent seeds per setting; fixed GPU assignment.
Source dcd3551c53a13c5613d8a8b644870de00dd57820. Descriptive95% intervals are clustered by workload seed.
Both windows are reported. The prespecified rule selected3s; heldout/crossover results remain separate.

| Case | TTL(s) | Pairs/seeds | Wall speedup [95% CI] | p95 TTFT speedup [95% CI] | Uncached input reduction [95% CI] |
|---|---:|---:|---:|---:|---:|
| ordinary-c8 | 3 | 3/3 | 1.009x [1.003, 1.015] | 0.992x [0.958, 1.039] | -0.53% [-1.58, +0.00] |
| trace-c16 | 3 | 3/3 | 1.024x [1.005, 1.039] | 0.991x [0.904, 1.041] | -0.10% [-10.32, +7.52] |
| trace-c8 | 3 | 3/3 | 1.007x [1.005, 1.010] | 1.067x [0.912, 1.302] | -2.04% [-6.12, +0.00] |
| ordinary-c8 | 30 | 3/3 | 1.000x [0.994, 1.007] | 0.984x [0.946, 1.051] | +0.00% [+0.00, +0.00] |
| trace-c16 | 30 | 3/3 | 1.008x [0.999, 1.024] | 0.986x [0.933, 1.032] | +10.22% [+1.02, +18.04] |
| trace-c8 | 30 | 3/3 | 0.998x [0.985, 1.010] | 1.061x [0.958, 1.175] | +3.18% [+0.00, +6.11] |

Uncached input is API prompt tokens minus cached tokens, excluding any additional recomputation after pauses. Logging instrumentation and scaled/capped waits affect absolute timings. Workload descriptors are per-request characteristics, not additional statistical repetitions.
