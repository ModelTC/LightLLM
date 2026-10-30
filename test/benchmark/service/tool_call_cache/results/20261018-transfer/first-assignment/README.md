# Complete first hardware assignment (interim)

These are **33 complete pairs**, including all 27 normal and 6 default-off comparisons in the first assignment, three workload seeds per case. Hardware and order crossover is still pending. This is not the final transfer result or a production benefit estimate. The later final 54+12 report includes these same pairs exactly once; never add this subset again.

Cache source, policy, benchmark, data and cases were frozen before this complete rerun. No independent performance tunes them. Public sanitized TraceLab metadata drives synthetic contexts and real generated tool-result loops; lengths/waits/output targets are explicitly scaled/capped. The source/hardware/configuration and exact report hashes are in [provenance](provenance.json). Requests, responses and trace identifiers are excluded.

## All normal cases

Ratios above 1 favor the enabled feature versus the same source with retention disabled. Intervals are descriptive 95% bootstrap intervals over three workload seeds, not independent requests.

| Case | Wall ratio [95% interval] | Uncached input reduction (%) | Agent mean request ratio | Ordinary one-shot p95 TTFT ratio |
|---|---:|---:|---:|---:|
| abandoned-agent25-c16 | 0.9844x [0.9785, 0.9931] | 0.00 | 1.1174x | 0.9504x |
| agent25-c16-fast | 1.0863x [1.0651, 1.1089] | 14.02 | 2.6022x | 0.9655x |
| agent50-c16-fast | 1.1292x [1.1211, 1.1381] | 21.80 | 1.9165x | 0.8786x |
| expired-agent25-c16 | 0.9690x [0.9542, 0.9774] | 0.69 | 0.9159x | 0.9236x |
| low-pressure-c4 | 1.0023x [1.0017, 1.0030] | 0.00 | n/a | n/a |
| ordinary-c8 | 0.9970x [0.9889, 1.0020] | -0.77 | n/a | n/a |
| output-shaped-agent50-c8 | 0.8838x [0.8542, 0.9045] | 11.32 | 1.0947x | 0.6992x |
| pressure-c24 | 1.2080x [1.1726, 1.2557] | 37.64 | n/a | n/a |
| realwait-trace-c16 | 1.0267x [1.0072, 1.0607] | 26.32 | n/a | n/a |

## Disabled feature versus original upstream

Control source `22888b4237aa2c895fc24176e08089cf14bc6c89`; both TTL 0 and identical physical capacity. Hardware/order reversal is pending.

| Case | Wall ratio [95% interval] | Server p95 TTFT ratio |
|---|---:|---:|
| ordinary-c8 | 1.0015x [0.9996, 1.0042] | 0.9888x |
| pressure-c24 | 1.0085x [0.9938, 1.0201] | 1.0067x |

The long-output case regresses to 0.884x wall despite lower uncached input; its ordinary mean request ratio is 0.8334x and ordinary p95 TTFT 0.6992x. Fast-agent gains coexist with ordinary tail costs (agent50 ordinary p95 TTFT 0.8786x). Expired and abandoned tool loops also regress. These are actual negative outcomes, not filtered exceptions. Keep the feature opt-in and do not extrapolate to all agent traffic.

[All normal aggregate/per-trial/cohort metrics](normal-public.json.gz) · [Normal workload distributions/caps/compactions](normal-workload.json.gz) · [All default-off metrics](defaultoff-public.json.gz) · [Default-off workload distributions](defaultoff-workload.json.gz).

Fixed plans and all 84 exploratory/validation pairs are [separate](../../20261018-rework/README.md). They are not pooled with this independent assessment.
