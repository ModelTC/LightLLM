"""CPU-only plan for safe per-request C4 next_n=2 packing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class C4RaggedPairPlan:
    kind: str
    q_lengths: Tuple[int, ...]
    packed_to_original: Tuple[int, ...]
    original_to_packed: Tuple[int, ...]
    packed_request: Tuple[int, ...]
    padded_rows: int


def build_c4_ragged_pair_plan(
    seq_lengths: Sequence[int], ready_lengths: Sequence[int], token_count: int, *, min_tokens: int = 256
) -> Optional[C4RaggedPairPlan]:
    """Build a request-local even-row plan, or retain the original N=1 path.

    Odd requests repeat only their final row.  Padding is limited to T/8, and
    pairs never cross request boundaries.
    """
    if type(token_count) is not int or token_count < min_tokens:
        return None
    if len(seq_lengths) == 0 or len(seq_lengths) != len(ready_lengths):
        return None
    if any(type(s) is not int or type(r) is not int for s, r in zip(seq_lengths, ready_lengths)):
        return None
    if any(s < 0 or r < 0 or r > s for s, r in zip(seq_lengths, ready_lengths)):
        return None
    q_lengths = tuple(s - r for s, r in zip(seq_lengths, ready_lengths))
    if sum(q_lengths) != token_count:
        return None
    padded_rows = sum(q_len & 1 for q_len in q_lengths)
    if padded_rows == 0:
        return C4RaggedPairPlan("all_even", q_lengths, (), (), (), 0)
    if padded_rows > token_count // 8:
        return None

    packed_to_original = []
    packed_request = []
    original_to_packed = [-1] * token_count
    original = 0
    for request, q_len in enumerate(q_lengths):
        for _ in range(q_len):
            original_to_packed[original] = len(packed_to_original)
            packed_to_original.append(original)
            packed_request.append(request)
            original += 1
        if q_len & 1:
            packed_to_original.append(original - 1)
            packed_request.append(request)
    if any(index < 0 for index in original_to_packed) or len(packed_to_original) & 1:
        return None
    if any(packed_request[i] != packed_request[i + 1] for i in range(0, len(packed_request), 2)):
        return None
    return C4RaggedPairPlan(
        "ragged",
        q_lengths,
        tuple(packed_to_original),
        tuple(original_to_packed),
        tuple(packed_request),
        padded_rows,
    )


def materialize_ragged_gpu_plan(plan: C4RaggedPairPlan, device):
    """Create retained pinned maps and enqueue their asynchronous H2D copies."""
    if plan.kind != "ragged":
        return None
    import torch

    def pinned(values):
        return torch.tensor(values, dtype=torch.int64, device="cpu", pin_memory=True)

    cpu = {
        "packed_to_original": pinned(plan.packed_to_original),
        "original_to_packed": pinned(plan.original_to_packed),
    }
    return {
        "cpu": cpu,
        "packed_to_original": cpu["packed_to_original"].to(device=device, non_blocking=True),
        "original_to_packed": cpu["original_to_packed"].to(device=device, non_blocking=True),
    }
