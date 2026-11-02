Controlled prefix residency; fixed hardware assignment; not online traffic throughput.

All paired inputs, outputs, end states and generated token counts matched. All GPU observations passed isolation and thermal checks. This fixed-assignment capacity-churn check establishes residency behavior, not production speedup.

| Case | Pairs | Candidate return cached tokens | Control return cached tokens |
|---|---:|---:|---:|
| expired | 5 | 0 | 0 |
| length | 5 | 0 | 0 |
| ordinary | 5 | 0 | 0 |
| partial | 5 | 8192 | 0 |
| tool | 5 | 5266 | 0 |

Initial and return prompt lengths, generated token counts, churn volume and individual timings are retained in the numerical JSON. Timings are diagnostic only.
