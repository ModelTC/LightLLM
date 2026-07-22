"""Asynchronous expert-row migration for EPLB."""
import os
import socket
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist

from lightllm.common.basemodel.triton_kernel.fused_moe.eplb_kernels import (
    eplb_push_copy,
)


@dataclass(frozen=True)
class TransferStep:
    dst_rank: int
    dst_slot: int
    src_rank: int
    src_local_row: int


def extract_expert_tensors(weight) -> List[Tuple[str, torch.Tensor]]:
    result = []
    for pack_name in ("w13", "w2"):
        pack = getattr(weight, pack_name)
        for value_name in ("weight", "weight_scale", "weight_zero_point"):
            tensor = getattr(pack, value_name, None)
            if tensor is not None:
                assert tensor.ndim >= 1 and tensor.is_contiguous(), f"{pack_name}.{value_name} must be contiguous"
                result.append((f"{pack_name}.{value_name}", tensor))
    return result


def commit_staging_rows(
    live: torch.Tensor,
    staging: torch.Tensor,
    experts_per_rank: int,
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
        live.narrow(0, experts_per_rank + run_start, run_length).copy_(
            staging.narrow(0, run_start, run_length), non_blocking=True
        )
        if dst_slot is not None:
            run_start = previous = dst_slot


def align_target_placement(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Canonicalize a target row layout without moving retained experts.

    EPLB placement is rank-based: redundant slots on one rank are
    interchangeable. Retained experts therefore keep their live physical
    slot, while new experts fill freed slots in the planner's target-row
    order. The returned placement is the single canonical layout that must
    be used both for transfers and for published routing metadata.
    """
    assert current.ndim == target.ndim == 2
    assert tuple(current.shape) == tuple(target.shape)

    current_rows = current.tolist()
    target_rows = target.tolist()
    aligned_target_rows = []
    for current_row, target_row in zip(current_rows, target_rows):
        current_slots = {expert: slot for slot, expert in enumerate(current_row)}
        target_experts = set(target_row)
        aligned_row = list(current_row)
        freed_slots = [slot for slot, expert in enumerate(current_row) if expert not in target_experts]
        new_experts = [expert for expert in target_row if expert not in current_slots]
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
    experts_per_rank = num_logical_experts // world_size
    current_rows = current.tolist()
    aligned_target_rows = align_target_placement(current, target).tolist()
    # A logical expert has one primary row and at most one redundant row per
    # rank, so this source list is already unique.  Build it once instead of
    # allocating/sorting a set for every destination slot.
    candidates_by_expert = [
        [(expert // experts_per_rank, expert % experts_per_rank)] for expert in range(num_logical_experts)
    ]
    for rank, row in enumerate(current_rows):
        for slot, expert in enumerate(row):
            candidates_by_expert[expert].append((rank, experts_per_rank + slot))
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


class _EPLBTransferBase:
    """Shared live/staging buffers and publish/commit lifecycle."""

    staging_depth = 1

    def __init__(self, weights, transfer_group, global_rank, world_size):
        self._eplb_states = [weight.expert_parallel_state.eplb for weight in weights]
        self.transfer_group = transfer_group
        self.global_rank = global_rank
        self.world_size = world_size
        self.experts_per_rank = weights[0].expert_parallel_state.primary_experts_per_rank
        self.device = weights[0].w13.weight.device
        self.live = [extract_expert_tensors(weight) for weight in weights]
        self._validate_live_layout(weights)
        redundant_slots = self._eplb_states[0].redundant_experts_per_rank
        self.staging = [
            [
                (
                    name,
                    torch.empty((redundant_slots,) + tuple(tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device),
                )
                for name, tensor in self.live[0]
            ]
            for _ in range(self.staging_depth)
        ]
        self._release = [threading.Event() for _ in range(self.staging_depth)]
        for release in self._release:
            release.set()
        self._error = None
        self._consumed_events = [torch.cuda.Event() for _ in range(self.staging_depth)]
        self._consumed_recorded = [False] * self.staging_depth
        self._changed_dst_slots = [()] * self.staging_depth
        self._pending = deque()
        self._pending_lock = threading.Lock()
        self._thread = None
        self._needs_staging_reuse_barrier = False

    def _validate_live_layout(self, weights) -> None:
        reference = [(name, tuple(tensor.shape[1:]), tensor.dtype, tensor.device) for name, tensor in self.live[0]]
        redundant_slots = self._eplb_states[0].redundant_experts_per_rank
        for layer_index, (state, tensors) in enumerate(zip(self._eplb_states, self.live)):
            layout = [(name, tuple(tensor.shape[1:]), tensor.dtype, tensor.device) for name, tensor in tensors]
            assert layout == reference, f"EPLB layer {layer_index} has incompatible expert tensor layout"
            assert state.redundant_experts_per_rank == redundant_slots, "EPLB redundant slot count must match"

    def _copy_layer(self, layer_index: int, plan: Sequence[TransferStep], staging) -> None:
        raise NotImplementedError

    def _copy_batch(self, batch) -> None:
        for layer_index, plan, _, staging in batch:
            self._copy_layer(layer_index, plan, staging)

    def _start_transfer_generation(self) -> None:
        """Prepare backend state after the in-flight worker check succeeds."""

    def _finish_transfer_generation(self) -> None:
        """Release backend state only after the migration worker has joined."""

    def start(self, layer_plans: Sequence[Tuple[int, Sequence[TransferStep]]]) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("EPLB transfer is already in flight")
        self._start_transfer_generation()
        self._error = None
        with self._pending_lock:
            self._pending.clear()

        def worker() -> None:
            try:
                torch.cuda.set_device(self.device)
                if not layer_plans:
                    self._finish_transfer_generation()
                for batch_start in range(0, len(layer_plans), self.staging_depth):
                    batch = []
                    for plan_index in range(batch_start, min(batch_start + self.staging_depth, len(layer_plans))):
                        layer_index, plan = layer_plans[plan_index]
                        buffer_index = plan_index % self.staging_depth
                        release = self._release[buffer_index]
                        # A buffer cannot be reused until its prior committed rows are no longer read by CUDA.
                        release.wait()
                        release.clear()
                        if self._consumed_recorded[buffer_index]:
                            self._consumed_events[buffer_index].synchronize()
                        self._changed_dst_slots[buffer_index] = tuple(
                            step.dst_slot for step in plan if step.dst_rank == self.global_rank
                        )
                        batch.append((layer_index, plan, buffer_index, self.staging[buffer_index]))
                    if batch_start > 0 and self._needs_staging_reuse_barrier:
                        # All destinations must finish consuming the prior IPC staging generation
                        # before a source can reuse the peer buffer for this batch.
                        dist.barrier(group=self.transfer_group)
                    self._copy_batch(batch)
                    if batch_start + self.staging_depth >= len(layer_plans):
                        self._finish_transfer_generation()
                    with self._pending_lock:
                        self._pending.extend((layer_index, buffer_index) for layer_index, _, buffer_index, _ in batch)
            except BaseException as exc:
                self._error = exc

        self._thread = threading.Thread(target=worker, name=f"eplb-{self.backend}", daemon=True)
        self._thread.start()

    def pending_layers(self):
        if self._error is not None:
            raise RuntimeError("EPLB migration worker failed") from self._error
        with self._pending_lock:
            return list(self._pending)

    def commit(self, layer_index: int, buffer_index: int, post_copy=None) -> None:
        with self._pending_lock:
            if not self._pending or self._pending[0] != (layer_index, buffer_index):
                raise RuntimeError("EPLB commit does not match the pending FIFO")
            self._pending.popleft()
            changed_dst_slots = self._changed_dst_slots[buffer_index]
        for (_, live), (_, staging) in zip(self.live[layer_index], self.staging[buffer_index]):
            commit_staging_rows(live, staging, self.experts_per_rank, changed_dst_slots)
        if post_copy is not None:
            post_copy()
        self._consumed_events[buffer_index].record(torch.cuda.current_stream())
        self._consumed_recorded[buffer_index] = True
        self._release[buffer_index].set()

    def finish(self) -> None:
        """Wait for the released migration worker to exit before another rebalance."""
        thread = self._thread
        if thread is None:
            return
        thread.join()
        self._thread = None
        if self._error is not None:
            raise RuntimeError("EPLB migration worker failed") from self._error


class NixlEPLBTransfer(_EPLBTransferBase):
    """GPU-direct UCX/NIXL EPLB transfer. Initialization errors are fatal."""

    backend = "nixl"
    staging_depth = 8
    _DEFAULT_UCX_TLS = "self,sm,cuda_ipc,cuda_copy,rc_x"

    def __init__(self, weights, transfer_group, global_rank, world_size):
        super().__init__(weights, transfer_group, global_rank, world_size)
        self._nixl_agent = None
        self._registered_descs = None
        self._remote_agents: Dict[int, str] = {}
        self._remote_layouts = {}
        self._xfer_cache = {}
        self._used_xfer_cache_keys = set()
        self._ipc_staging = {}
        self._same_node_ranks = set()
        self._cross_node_ranks = set()
        self._push_stream = torch.cuda.Stream(device=self.device)
        self._push_descriptor_cache = {}
        self._used_push_descriptor_cache_keys = set()
        try:
            self._init_ipc_metadata()
            if self._cross_node_ranks:
                os.environ.setdefault("UCX_TLS", self._DEFAULT_UCX_TLS)
                try:
                    import nixl
                except Exception as exc:
                    raise RuntimeError("NIXL EPLB backend requires the nixl package for cross-node transfer") from exc
                agent_name = f"lightllm-eplb-{socket.gethostname()}-{os.getpid()}-rank-{global_rank}"
                config = nixl.nixl_agent_config(enable_prog_thread=True, enable_listen_thread=False, backends=["UCX"])
                self._nixl_agent = nixl.nixl_agent(agent_name, config)
                reg_tensors = [tensor for layer in self.live for _, tensor in layer] + [
                    tensor for staging in self.staging for _, tensor in staging
                ]
                self._registered_descs = self._nixl_agent.get_reg_descs(reg_tensors)
                self._nixl_agent.register_memory(self._registered_descs, backends=["UCX"])
                self._init_remote_metadata()
        except Exception as exc:
            self.shutdown()
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError("NIXL EPLB initialization failed") from exc

    def _local_layout(self):
        return [
            [(name, tensor.data_ptr(), tensor.get_device(), tensor[0].nbytes) for name, tensor in layer]
            for layer in self.live
        ]

    def _init_ipc_metadata(self) -> None:
        hostnames = [None] * self.world_size
        dist.all_gather_object(hostnames, socket.gethostname(), group=self.transfer_group)
        local_hostname = hostnames[self.global_rank]
        self._needs_staging_reuse_barrier = len(set(hostnames)) < len(hostnames)
        self._same_node_ranks = {rank for rank, hostname in enumerate(hostnames) if hostname == local_hostname}
        self._cross_node_ranks = set(range(self.world_size)) - self._same_node_ranks
        for layer in self.live:
            for name, tensor in layer:
                if name.endswith(".weight") and tensor[0].nbytes % 16:
                    raise RuntimeError(f"NIXL source-push requires 16-byte aligned weight rows: {name}")

        from lightllm.server.router.model_infer.mode_backend.pd.p2p_fix import (
            p2p_fix_rebuild_cuda_tensor,
            reduce_tensor,
        )

        exports = {}
        for target_rank in self._same_node_ranks - {self.global_rank}:
            exports[target_rank] = {
                "staging": [
                    [(name, tuple(tensor.shape), tensor.dtype, reduce_tensor(tensor)[1]) for name, tensor in staging]
                    for staging in self.staging
                ],
            }
        all_exports = [None] * self.world_size
        dist.all_gather_object(all_exports, exports, group=self.transfer_group)

        torch.cuda.set_device(self.device)
        for dst_rank in self._same_node_ranks - {self.global_rank}:
            metadata = all_exports[dst_rank].get(self.global_rank)
            if metadata is None or len(metadata["staging"]) != self.staging_depth:
                raise RuntimeError(f"NIXL IPC destination rank {dst_rank} has incompatible staging metadata")
            rebuilt_staging = []
            for remote_staging, local_staging in zip(metadata["staging"], self.staging):
                if len(remote_staging) != len(local_staging):
                    raise RuntimeError(f"NIXL IPC destination rank {dst_rank} staging tensor count mismatch")
                rebuilt = []
                for (name, shape, dtype, args), (local_name, local_tensor) in zip(remote_staging, local_staging):
                    if name != local_name or shape != tuple(local_tensor.shape) or dtype != local_tensor.dtype:
                        raise RuntimeError(f"NIXL IPC destination rank {dst_rank} staging layout mismatch for {name}")
                    tensor = p2p_fix_rebuild_cuda_tensor(*args)
                    if tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.device != local_tensor.device:
                        raise RuntimeError(
                            f"NIXL IPC destination rank {dst_rank} staging rebuild validation failed for {name}"
                        )
                    rebuilt.append((name, tensor))
                rebuilt_staging.append(rebuilt)
            self._ipc_staging[dst_rank] = rebuilt_staging

    def _init_remote_metadata(self) -> None:
        metadata = self._nixl_agent.get_agent_metadata()
        all_metadata = [None] * self.world_size
        all_layouts = [None] * self.world_size
        dist.all_gather_object(all_metadata, metadata, group=self.transfer_group)
        dist.all_gather_object(all_layouts, self._local_layout(), group=self.transfer_group)
        for rank in self._cross_node_ranks:
            layout = all_layouts[rank]
            if len(layout) != len(self.live):
                raise RuntimeError(f"NIXL remote rank {rank} has incompatible layer layout")
            self._remote_agents[rank] = self._nixl_agent.add_remote_agent(all_metadata[rank])
            self._remote_layouts[rank] = layout

    def _wait_xfers(self, xfers) -> None:
        pending = []
        for item in xfers:
            state = self._nixl_agent.transfer(item[2])
            if state == "ERR":
                raise RuntimeError("NIXL READ post failed")
            if state == "PROC":
                pending.append(item)
        while pending:
            remaining = []
            for item in pending:
                state = self._nixl_agent.check_xfer_state(item[2])
                if state == "ERR":
                    raise RuntimeError("NIXL READ transfer failed")
                if state != "DONE":
                    remaining.append(item)
            pending = remaining

    def _release_xfers(self, xfers) -> None:
        unreleased = []
        errors = []
        for local_dlist, remote_dlist, xfer in xfers:
            remaining = [local_dlist, remote_dlist, xfer]
            for remaining_index, handle, release in (
                (2, xfer, self._nixl_agent.release_xfer_handle),
                (1, remote_dlist, self._nixl_agent.release_dlist_handle),
                (0, local_dlist, self._nixl_agent.release_dlist_handle),
            ):
                if handle is not None:
                    try:
                        release(handle)
                    except Exception as exc:
                        errors.append(exc)
                    else:
                        remaining[remaining_index] = None
            if any(handle is not None for handle in remaining):
                unreleased.append(tuple(remaining))
        if errors:
            error = RuntimeError("NIXL transfer handle release failed")
            error.unreleased_xfers = unreleased
            raise error from errors[0]

    @staticmethod
    def _contiguous_runs(steps):
        ordered = sorted(steps, key=lambda step: (step.src_local_row, step.dst_slot))
        runs = []
        for step in ordered:
            if (
                runs
                and step.src_local_row == runs[-1][-1].src_local_row + 1
                and step.dst_slot == runs[-1][-1].dst_slot + 1
            ):
                runs[-1].append(step)
            else:
                runs.append([step])
        return runs

    @staticmethod
    def _remote_read_cache_key(src_rank: int, entries):
        return (
            src_rank,
            tuple(
                (
                    layer_index,
                    tuple((step.src_local_row, step.dst_slot) for step in run),
                    tuple(tensor.data_ptr() for _, tensor in staging),
                )
                for layer_index, run, staging in entries
            ),
        )

    def _push_staging(self, dst_rank: int, buffer_index: int):
        return self.staging[buffer_index] if dst_rank == self.global_rank else self._ipc_staging[dst_rank][buffer_index]

    def _cached_descriptor_tensors(self, copies):
        key = tuple((source.data_ptr(), destination.data_ptr()) for destination, source in copies)
        cached = self._push_descriptor_cache.get(key)
        if cached is None:
            src_ptrs = torch.tensor([source.data_ptr() for _, source in copies], dtype=torch.int64, device=self.device)
            dst_ptrs = torch.tensor(
                [destination.data_ptr() for destination, _ in copies], dtype=torch.int64, device=self.device
            )
            cached = (src_ptrs, dst_ptrs)
            self._push_descriptor_cache[key] = cached
        self._used_push_descriptor_cache_keys.add(key)
        return cached

    def _push_same_node(self, dst_rank: int, entries) -> None:
        staging_by_buffer = {buffer_index: self._push_staging(dst_rank, buffer_index) for _, _, buffer_index in entries}
        weight_groups = defaultdict(list)
        small_copies = []
        for layer_index, run, buffer_index in entries:
            staging = staging_by_buffer[buffer_index]
            source_layer = self.live[layer_index]
            first = run[0]
            run_len = len(run)
            for (name, source_tensor), (staging_name, staging_tensor) in zip(source_layer, staging):
                if name != staging_name:
                    raise RuntimeError("NIXL source-push staging tensor name mismatch")
                source_rows = source_tensor.narrow(0, first.src_local_row, run_len)
                destination_rows = staging_tensor.narrow(0, first.dst_slot, run_len)
                if name.endswith(".weight"):
                    if destination_rows.nbytes % 16:
                        raise RuntimeError(f"NIXL source-push requires 16-byte aligned weight rows: {name}")
                    weight_groups[destination_rows.nbytes].append((destination_rows, source_rows))
                else:
                    small_copies.append((destination_rows, source_rows))
        with torch.cuda.stream(self._push_stream):
            for nbytes, copies in weight_groups.items():
                src_ptrs, dst_ptrs = self._cached_descriptor_tensors(copies)
                eplb_push_copy(src_ptrs, dst_ptrs, nbytes)
            for destination_rows, source_rows in small_copies:
                destination_rows.copy_(source_rows, non_blocking=True)

    def _get_remote_read(self, src_rank: int, entries):
        cache_key = self._remote_read_cache_key(src_rank, entries)
        cached = self._xfer_cache.get(cache_key)
        if cached is not None:
            self._used_xfer_cache_keys.add(cache_key)
            return cached
        local_descs = []
        remote_descs = []
        local_dlist = remote_dlist = xfer = None
        try:
            for layer_index, run, staging in entries:
                remote_layer = self._remote_layouts[src_rank][layer_index]
                if len(remote_layer) != len(staging):
                    raise RuntimeError(f"NIXL remote rank {src_rank} has incompatible layer layout")
                first = run[0]
                run_len = len(run)
                for tensor_index, (_, staging_tensor) in enumerate(staging):
                    name, remote_ptr, remote_device, remote_nbytes = remote_layer[tensor_index]
                    if (
                        name != self.live[layer_index][tensor_index][0]
                        or remote_nbytes != staging_tensor[first.dst_slot].nbytes
                    ):
                        raise RuntimeError(f"NIXL remote rank {src_rank} descriptor range mismatch")
                    local_descs.append(
                        (
                            staging_tensor[first.dst_slot].data_ptr(),
                            run_len * remote_nbytes,
                            staging_tensor.get_device(),
                        )
                    )
                    remote_descs.append(
                        (remote_ptr + first.src_local_row * remote_nbytes, run_len * remote_nbytes, remote_device)
                    )
            local_dlist = self._nixl_agent.prep_xfer_dlist(
                "NIXL_INIT_AGENT", self._nixl_agent.get_xfer_descs(local_descs, "VRAM"), backends=["UCX"]
            )
            remote_dlist = self._nixl_agent.prep_xfer_dlist(
                self._remote_agents[src_rank], self._nixl_agent.get_xfer_descs(remote_descs, "VRAM"), backends=["UCX"]
            )
            xfer = self._nixl_agent.make_prepped_xfer(
                "READ",
                local_dlist,
                list(range(len(local_descs))),
                remote_dlist,
                list(range(len(remote_descs))),
                backends=["UCX"],
            )
            selected_backend = self._nixl_agent.query_xfer_backend(xfer)
            if selected_backend != "UCX":
                raise RuntimeError("NIXL EPLB READ did not select UCX")
            self._xfer_cache[cache_key] = (local_dlist, remote_dlist, xfer)
            self._used_xfer_cache_keys.add(cache_key)
            return self._xfer_cache[cache_key]
        except Exception:
            self._release_xfers([(local_dlist, remote_dlist, xfer)])
            raise

    def _copy_batch(self, batch) -> None:
        remote_entries = defaultdict(list)
        push_entries = defaultdict(list)
        for layer_index, plan, _, staging in batch:
            steps_by_source = defaultdict(list)
            for step in plan:
                if step.dst_rank == self.global_rank:
                    steps_by_source[step.src_rank].append(step)
            for src_rank, steps in steps_by_source.items():
                entries = [(layer_index, run, staging) for run in self._contiguous_runs(steps)]
                if src_rank not in self._same_node_ranks:
                    remote_entries[src_rank].extend(entries)
        # Source rank owns node-local copies.  All ranks build the same batch,
        # so buffer_index is the receiver's staging depth index on every peer.
        for layer_index, plan, buffer_index, _ in batch:
            by_destination = defaultdict(list)
            for step in plan:
                if step.src_rank == self.global_rank and step.dst_rank in self._same_node_ranks:
                    by_destination[step.dst_rank].append(step)
            for dst_rank, steps in by_destination.items():
                push_entries[dst_rank].extend((layer_index, run, buffer_index) for run in self._contiguous_runs(steps))
        for dst_rank, entries in push_entries.items():
            self._push_same_node(dst_rank, entries)

        xfers = [self._get_remote_read(src_rank, entries) for src_rank, entries in remote_entries.items()]
        self._wait_xfers(xfers)
        self._push_stream.synchronize()
        # Before a rank publishes this batch it has completed its outgoing source-pushes and
        # incoming UCX READs. The manager's global MIN-ready gate therefore means all transfers
        # are complete before any rank commits, without a destination-side GPU wait.

    def _start_transfer_generation(self) -> None:
        self._used_xfer_cache_keys.clear()
        self._used_push_descriptor_cache_keys.clear()

    def _finish_transfer_generation(self) -> None:
        errors = []
        for cache_key in set(self._xfer_cache) - self._used_xfer_cache_keys:
            xfer = self._xfer_cache[cache_key]
            try:
                self._release_xfers([xfer])
            except Exception as exc:
                unreleased = getattr(exc, "unreleased_xfers", None)
                if unreleased:
                    self._xfer_cache[cache_key] = unreleased[0]
                errors.append(exc)
            else:
                del self._xfer_cache[cache_key]
        for cache_key in set(self._push_descriptor_cache) - self._used_push_descriptor_cache_keys:
            del self._push_descriptor_cache[cache_key]
        if errors:
            raise RuntimeError("NIXL EPLB cache eviction failed") from errors[0]

    def shutdown(self) -> None:
        agent = self._nixl_agent
        errors = []
        getattr(self, "_used_xfer_cache_keys", set()).clear()
        getattr(self, "_used_push_descriptor_cache_keys", set()).clear()
        if agent is not None:
            for cache_key, xfer in list(self._xfer_cache.items()):
                try:
                    self._release_xfers([xfer])
                except Exception as exc:
                    unreleased = getattr(exc, "unreleased_xfers", None)
                    if unreleased:
                        self._xfer_cache[cache_key] = unreleased[0]
                    errors.append(exc)
                else:
                    del self._xfer_cache[cache_key]
        if errors:
            raise RuntimeError("NIXL EPLB shutdown failed") from errors[0]
        for remote_name in list(self._remote_agents.values()):
            if agent is not None:
                try:
                    agent.remove_remote_agent(remote_name)
                except Exception as exc:
                    errors.append(exc)
        self._remote_agents.clear()
        self._remote_layouts.clear()
        if agent is not None and self._registered_descs is not None:
            try:
                agent.deregister_memory(self._registered_descs, backends=["UCX"])
            except Exception as exc:
                errors.append(exc)
        self._registered_descs = None
        self._nixl_agent = None
        getattr(self, "_ipc_staging", {}).clear()
        getattr(self, "_push_descriptor_cache", {}).clear()
        if errors:
            raise RuntimeError("NIXL EPLB shutdown failed") from errors[0]

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
