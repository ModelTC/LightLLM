"""Layer-by-layer expert-row migration for EPLB."""

import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.distributed as dist

from lightllm.common.eplb_utils import extract_eplb_expert_tensors


@dataclass(frozen=True)
class TransferStep:
    dst_rank: int
    dst_slot: int
    src_rank: int
    src_local_row: int


def align_target_placement(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Canonicalize a target row layout without moving retained experts."""
    assert current.ndim == target.ndim == 2
    assert tuple(current.shape) == tuple(target.shape)

    aligned_target_rows = []
    for current_row, target_row in zip(current.tolist(), target.tolist()):
        remaining_target = Counter(target_row)
        aligned_row = list(current_row)
        freed_slots = []
        for slot, expert in enumerate(current_row):
            if remaining_target[expert] > 0:
                remaining_target[expert] -= 1
            else:
                freed_slots.append(slot)
        new_experts = []
        for expert in target_row:
            if remaining_target[expert] > 0:
                new_experts.append(expert)
                remaining_target[expert] -= 1
        assert len(freed_slots) == len(new_experts)
        for slot, expert in zip(freed_slots, new_experts):
            aligned_row[slot] = expert
        aligned_target_rows.append(aligned_row)
    return target.new_tensor(aligned_target_rows)


def build_transfer_plan(
    current: torch.Tensor,
    target: torch.Tensor,
    num_logical_experts: int,
    world_size: int,
    node_world_size: int,
) -> List[TransferStep]:
    assert tuple(current.shape) == tuple(target.shape) == (world_size, current.shape[1])
    num_experts_per_rank = num_logical_experts // world_size
    current_rows = current.tolist()
    aligned_target_rows = align_target_placement(current, target).tolist()
    # A primary row is always a valid source. Existing replicas are also
    # candidates so that a destination can prefer a same-node copy.
    candidates_by_expert = [
        [(expert // num_experts_per_rank, expert % num_experts_per_rank)] for expert in range(num_logical_experts)
    ]
    for rank, row in enumerate(current_rows):
        for slot, expert in enumerate(row):
            candidates_by_expert[expert].append((rank, num_experts_per_rank + slot))
    source_load = [0] * world_size
    plan = []
    for dst_rank in range(world_size):
        for dst_slot, expert in enumerate(aligned_target_rows[dst_rank]):
            if expert == current_rows[dst_rank][dst_slot]:
                continue
            src_rank, src_row = min(
                candidates_by_expert[expert],
                key=lambda item: (
                    item[0] // node_world_size != dst_rank // node_world_size,
                    source_load[item[0]],
                    item[0],
                    item[1],
                ),
            )
            source_load[src_rank] += 1
            plan.append(TransferStep(dst_rank, dst_slot, src_rank, src_row))
    return plan


class PinnedMemoryEPLBTransfer:
    """Move one layer at a time through reusable pinned CPU row buffers.

    Every rank executes the same ordered Gloo broadcasts. A source first
    copies a live GPU row to pinned memory; destinations then copy that row
    into a single-layer GPU staging buffer. The inference thread publishes
    the staging rows and routing metadata together at a safe forward boundary.
    """

    def __init__(self, weights, transfer_group, global_rank):
        self._eplb_impls = [weight.fuse_moe_impl for weight in weights]
        self.transfer_group = transfer_group
        self.global_rank = global_rank
        self.num_experts_per_rank = self._eplb_impls[0].num_primary_experts_per_rank
        self.device = weights[0].w13.weight.device
        self.live = [extract_eplb_expert_tensors(weight) for weight in weights]
        self._validate_live_layout()

        num_redundant_slots = self._eplb_impls[0].num_redundant_experts_per_rank
        self.staging = [
            (
                name,
                torch.empty(
                    (num_redundant_slots,) + tuple(tensor.shape[1:]),
                    dtype=tensor.dtype,
                    device=tensor.device,
                ),
            )
            for name, tensor in self.live[0]
        ]
        self.pinned_rows = [
            (
                name,
                torch.empty(
                    tuple(tensor.shape[1:]),
                    dtype=tensor.dtype,
                    device="cpu",
                    pin_memory=True,
                ),
            )
            for name, tensor in self.live[0]
        ]
        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._release = threading.Event()
        self._release.set()
        self._consumed_event = torch.cuda.Event()
        self._consumed_recorded = False
        self._ready = None
        self._ready_lock = threading.Lock()
        self._error = None
        self._thread = None

    def _validate_live_layout(self) -> None:
        reference = [(name, tuple(tensor.shape[1:]), tensor.dtype, tensor.device) for name, tensor in self.live[0]]
        num_redundant_slots = self._eplb_impls[0].num_redundant_experts_per_rank
        for layer_index, (impl, tensors) in enumerate(zip(self._eplb_impls, self.live)):
            layout = [(name, tuple(tensor.shape[1:]), tensor.dtype, tensor.device) for name, tensor in tensors]
            assert layout == reference, f"EPLB layer {layer_index} has incompatible expert tensor layout"
            assert impl.num_redundant_experts_per_rank == num_redundant_slots, "EPLB redundant slot count must match"

    @staticmethod
    def _group_steps_by_source(plan: Sequence[TransferStep]):
        grouped = defaultdict(list)
        for step in plan:
            grouped[(step.src_rank, step.src_local_row)].append(step)
        return [(source, grouped[source]) for source in sorted(grouped)]

    def _copy_layer(self, layer_index: int, plan: Sequence[TransferStep]) -> None:
        live_tensors = self.live[layer_index]
        for (src_rank, src_local_row), steps in self._group_steps_by_source(plan):
            if self.global_rank == src_rank:
                with torch.cuda.stream(self._copy_stream):
                    for (_, live), (_, pinned) in zip(live_tensors, self.pinned_rows):
                        pinned.copy_(live[src_local_row], non_blocking=True)
            # Gloo must not read the CPU row before the device-to-host copy completes.
            self._copy_stream.synchronize()
            for _, pinned in self.pinned_rows:
                dist.broadcast(pinned, src=src_rank, group=self.transfer_group)

            dst_slots = sorted({step.dst_slot for step in steps if step.dst_rank == self.global_rank})
            if dst_slots:
                with torch.cuda.stream(self._copy_stream):
                    for (_, staging), (_, pinned) in zip(self.staging, self.pinned_rows):
                        for dst_slot in dst_slots:
                            staging[dst_slot].copy_(pinned, non_blocking=True)
        self._copy_stream.synchronize()

    def start(self, layer_plans: Sequence[Tuple[int, Sequence[TransferStep]]]) -> None:
        if self._thread is not None:
            raise RuntimeError("EPLB transfer has not been finished")
        self._error = None
        with self._ready_lock:
            if self._ready is not None:
                raise RuntimeError("EPLB ready layer has not been committed")

        def worker() -> None:
            try:
                torch.cuda.set_device(self.device)
                for layer_index, plan in layer_plans:
                    self._release.wait()
                    self._release.clear()
                    if self._consumed_recorded:
                        self._consumed_event.synchronize()
                    changed_dst_slots = tuple(
                        sorted({step.dst_slot for step in plan if step.dst_rank == self.global_rank})
                    )
                    self._copy_layer(layer_index, plan)
                    with self._ready_lock:
                        self._ready = (layer_index, changed_dst_slots)
            except BaseException as exc:
                self._error = exc

        self._thread = threading.Thread(target=worker, name="eplb-pin-memory", daemon=True)
        self._thread.start()

    def ready_layer(self):
        if self._error is not None:
            raise RuntimeError("EPLB migration worker failed") from self._error
        with self._ready_lock:
            return None if self._ready is None else self._ready[0]

    def commit(self, layer_index: int, post_copy=None) -> None:
        with self._ready_lock:
            if self._ready is None or self._ready[0] != layer_index:
                raise RuntimeError("EPLB commit does not match the ready layer")
            _, changed_dst_slots = self._ready
            self._ready = None
        for (_, live), (_, staging) in zip(self.live[layer_index], self.staging):
            _commit_staging_rows(live, staging, self.num_experts_per_rank, changed_dst_slots)
        if post_copy is not None:
            post_copy()
        self._consumed_event.record(torch.cuda.current_stream())
        self._consumed_recorded = True
        self._release.set()

    def finish(self) -> None:
        thread = self._thread
        if thread is None:
            return
        thread.join()
        self._thread = None
        if self._error is not None:
            raise RuntimeError("EPLB migration worker failed") from self._error


def _commit_staging_rows(
    live: torch.Tensor,
    staging: torch.Tensor,
    num_experts_per_rank: int,
    changed_dst_slots: Sequence[int],
) -> None:
    slots = sorted(set(changed_dst_slots))
    if not slots:
        return
    run_start = previous = slots[0]
    for dst_slot in (*slots[1:], None):
        if dst_slot is not None and dst_slot == previous + 1:
            previous = dst_slot
            continue
        run_length = previous - run_start + 1
        live.narrow(0, num_experts_per_rank + run_start, run_length).copy_(
            staging.narrow(0, run_start, run_length), non_blocking=True
        )
        if dst_slot is not None:
            run_start = previous = dst_slot
