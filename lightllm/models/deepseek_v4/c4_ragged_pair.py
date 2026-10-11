"""CPU-only plan for safe per-request C4 next_n=2 packing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass(frozen=True)
class C4RaggedPairPlan:
    kind: str
    packed_to_original: Tuple[int, ...]
    original_to_packed: Tuple[int, ...]


def build_c4_ragged_pair_plan(
    seq_lengths: Sequence[int], ready_lengths: Sequence[int], token_count: int
) -> C4RaggedPairPlan:
    """Build a request-local even-row N=2 plan. Odd requests repeat their final row."""
    if type(token_count) is not int or token_count < 0:
        raise ValueError("token_count must be a non-negative int")
    if len(seq_lengths) != len(ready_lengths):
        raise ValueError("C4 pair plan requires matched request lengths")
    if any(type(s) is not int or type(r) is not int for s, r in zip(seq_lengths, ready_lengths)):
        raise ValueError("C4 pair plan lengths must be ints")
    if any(s < 0 or r < 0 or r > s for s, r in zip(seq_lengths, ready_lengths)):
        raise ValueError("C4 pair plan has invalid ready/sequence lengths")
    q_lengths = tuple(s - r for s, r in zip(seq_lengths, ready_lengths))
    if sum(q_lengths) != token_count:
        raise ValueError("C4 pair plan token count does not match request lengths")
    padded_rows = sum(q_len & 1 for q_len in q_lengths)
    if padded_rows == 0:
        return C4RaggedPairPlan("all_even", (), ())
    packed_to_original = []
    original_to_packed = [-1] * token_count
    original = 0
    for q_len in q_lengths:
        for _ in range(q_len):
            original_to_packed[original] = len(packed_to_original)
            packed_to_original.append(original)
            original += 1
        if q_len & 1:
            packed_to_original.append(original - 1)
    # Input validation and construction guarantee every original row is mapped once,
    # packed rows are even, and odd padding repeats only that request's final row.
    return C4RaggedPairPlan("ragged", tuple(packed_to_original), tuple(original_to_packed))


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
