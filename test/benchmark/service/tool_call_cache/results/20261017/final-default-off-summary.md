Final-source Qwen3-4B default-disabled versus unchanged upstream22888b4237aa2c895fc24176e08089cf14bc6c89; all20 pairs,2 prespecified cases,5 independent seeds and hardware crossover. Candidate dcd3551c53a13c5613d8a8b644870de00dd57820 TTL0/fraction0.5. Primary candidate GPU0/control GPU1, crossover candidate GPU1/control GPU0. Same-seed assignments clustered. Equal debug telemetry; finite low-seed study does not establish zero overhead in all deployments.

Paired ratios; 95% descriptive bootstrap intervals clustered by workload seed; 10000 resamples.

| Case | TTL(s) | Pairs/seeds | Wall speedup [95% CI] | p95 TTFT speedup [95% CI] | Uncached input reduction [95% CI] |
|---|---:|---:|---:|---:|---:|
| ordinary-c8 | 0 | 10/5 | 1.000x [0.999, 1.002] | 0.966x [0.905, 1.031] | -0.75% [-2.25, +0.00] |
| trace-c16 | 0 | 10/5 | 0.999x [0.995, 1.003] | 0.953x [0.905, 0.994] | -1.45% [-4.45, +1.33] |

Uncached input = API prompt tokens minus cached tokens, excluding additional recomputation after pauses. Absolute timings include equal debug telemetry and the declared scaled/capped tool waits. Per-request workload descriptors are not additional independent repetitions. All audited groups and all paired trials are retained.
