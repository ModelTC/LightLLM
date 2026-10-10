# ReplaySSM for GDN and KDA

ReplaySSM reduces speculative SSM storage by keeping one checkpoint per request
and replaying accepted updates. It is opt-in; `--ssm_state_mode native` remains
the default.

## Configuration

```bash
LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 python -m lightllm.server.api_server \
    --model_dir /path/to/qwen3.5 \
    --ssm_state_mode replay \
    --linear_att_ssm_data_type bfloat16 \
    --replayssm_cache_len 8 \
    --replayssm_projection_mode precompute
```

The existing MTP flags can be added to this command. History capacity must cover
the maximum verify width (`mtp_step + 1`), including dynamic MTP. Supported
capacities are 4, 8, 16, 32 and 64.

| State mode | Storage and update policy |
| --- | --- |
| `native` | Existing per-token speculative state snapshots. |
| `compact` | One state plus this round's raw inputs; merge the accepted prefix every round. Requires MTP. |
| `replay` | One state plus bounded accepted history; fold before the next verify window would overflow. Supports ordinary decode and MTP. |

Non-native modes support Qwen3-Next/Qwen3.5 GDN and GLM-5.3-FLASH KDA with FP32 or BF16
state. Other models retain their native path. `--replayssm_projection_mode`
selects `inline` (default) or `precompute`; the latter is only valid with replay.
Choose the projection mode and capacity using the target workload.
Without MTP, replay adds history storage to the single native checkpoint.

## State lifecycle and precision

Verify overwrites the uncommitted history suffix. Acceptance advances only the
accepted prefix, so rejected proposals never enter a checkpoint. Fold alternates
history buffers to avoid overwriting records still read by another CTA. Padding
and the HOLD request do not advance history.

Prefill folds accepted updates before using the existing chunk kernel. CPU
prefix checkpoints and PD pages contain canonical SSM state and the accepted
conv window, without replay scratch. Exporting a checkpoint does not mutate
the active state. Restoring a checkpoint or reusing a request slot invalidates
history and resets the MTP offset.

Replay accumulates in FP32 and rounds when folding or materializing a checkpoint.
BF16 native and compact modes round after each token. Replay therefore does not
promise bitwise equivalence or identical generated text to native BF16. The
`precompute` path also uses activation-precision projections. Correctness tests
use an independent recurrence with each mode's rounding boundary.

The SSM autotuner measures prepare, forward, acceptance and fold on disposable
state; serving state is never a tuning input. Each batch bucket selects its own
layout before CUDA Graph capture. Model outputs carry that layout through
unpadding, and compact acceptance uses the corresponding verify layout for each
microbatch. Replay checkpoint reconstruction keeps a batch-independent layout.
Cached configurations distinguish token count, sequence capacity, precision,
stride, heads, verify width, history capacity, projection mode and recurrent rule.

KDA uses a per-key-dimension log decay, while GDN uses one scalar per value
head. KDA history stores the full gate vector; checkpoint projection applies
the cumulative vector decay to Q/K before projecting. The low-rank correction
uses forward decay between accepted tokens, without inverse exponential
scaling. The model's `gate_lower_bound` also separates autotune configurations.
Accepted-state exports retain GLM's indexer tail for PD; CPU prefix restore
clears the tail at aligned pool boundaries.

## Memory and performance

For a TP4 model with 48 GDN layers, 12 local value heads and 128x128 BF16 state,
one full-model checkpoint occupies 18 MiB per GPU. Native MTP3 uses 72 MiB per
request. Replay L8/precompute uses about 24.07 MiB including history and cursors.
Conv state, full-attention KV, request tables and sampling buffers are additional
allocations.

For GLM-5.3-FLASH TP8 (34 KDA layers, 8 local heads), one FP32 checkpoint
is 17 MiB per GPU. Native MTP2 uses 51 MiB per request; replay L8/inline
uses about 25.52 MiB including history. With BF16 state the corresponding
figures are 25.5 and 17.02 MiB.

More KV space improves throughput only when the workload and request capacity
can use it. Increasing request capacity can also increase per-token latency.
Keep algorithm comparisons at equal capacity separate from capacity comparisons.

## Tests

```bash
exp -m "ReplaySSM GDN/KDA lifecycle regression" /usr/bin/env PYTHONPATH=. python -m pytest -q \
    unit_tests/server/test_ssm_state_mode.py \
    unit_tests/common/basemodel/attention/linear/test_gdn.py \
    unit_tests/common/basemodel/triton_kernel/linear_att/test_replayssm*.py \
    unit_tests/common/basemodel/triton_kernel/linear_att/test_ssm_autotune.py \
    unit_tests/common/basemodel/triton_kernel/linear_att/test_mtp_state_params.py \
    unit_tests/common/basemodel/triton_kernel/linear_att/test_acceptance_fusion.py \
    unit_tests/common/test_paged_kv_transfer.py \
    unit_tests/models/glm5_next/test_{mtp,cache,pd_cache}.py
```

CUDA tests cover partial acceptance, history folds, BF16 rounding, grouped keys,
dynamic verify lengths, CUDA Graph replay, HOLD padding, checkpoint/PD transfer,
slot reuse and fused acceptance. CPU tests cover startup validation and tuner
isolation. CUDA tests skip when a GPU is unavailable.
