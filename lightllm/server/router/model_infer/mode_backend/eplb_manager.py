import os
import threading
import time
from pathlib import Path
from typing import Dict
from collections import deque

import torch
import torch.distributed as dist

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.eplb_utils import find_ep_moe_weights
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_placement import (
    build_logical_to_physical_maps_for_layers,
    select_improving_placements,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_planner import (
    plan_eplb_candidate,
)
from lightllm.server.router.model_infer.mode_backend.eplb_transfer import (
    NixlEPLBTransfer,
    align_target_placement,
    build_transfer_plan,
)
from lightllm.utils.dist_utils import get_global_rank, get_global_world_size, get_node_world_size
from lightllm.common.eplb_planner_worker import EPLBPlannerWorker
from lightllm.common.eplb_utils import EPLB_STEADY_SAMPLE_STEPS
from lightllm.utils.envs_utils import (
    get_eplb_placement_stickiness,
    get_eplb_planner_process,
    get_eplb_rebalance_gain_threshold,
    get_prefill_eplb_step_interval,
    get_prefill_eplb_steady_sample_steps,
    get_prefill_eplb_min_rebalance_interval,
)
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)
EPLB_MIN_AVG_TOKENS_PER_EXPERT = 100
EPLB_EXPERT_ALIGNMENT = 128
EPLB_CONTROL_ERROR = -1


class EPLBManager:
    """Online EPLB with asynchronous GPU expert migration."""

    def __init__(self, model: TpPartBaseModel):
        self.weights = find_ep_moe_weights(model)
        assert self.weights, "EPLB requires at least one EP MoE layer"
        self.global_rank = get_global_rank()
        self.world_size = get_global_world_size()
        self.node_world_size = get_node_world_size()
        self._eplb_states = [weight.expert_parallel_state.eplb for weight in self.weights]
        self.step_interval = get_prefill_eplb_step_interval()
        self.rebalance_gain_threshold = get_eplb_rebalance_gain_threshold()
        self.placement_stickiness = get_eplb_placement_stickiness()
        self.steady_sample_steps = get_prefill_eplb_steady_sample_steps()
        self.min_rebalance_interval = get_prefill_eplb_min_rebalance_interval()
        self.sampling_interval = self.step_interval
        self.prefill_steps = 0
        self._initial_plan_complete = False
        self._use_full_window = True
        self._next_evaluation_step = self.step_interval
        self._last_evaluation_step = 0
        self._sample_step_counts = deque([0], maxlen=max(self.step_interval, EPLB_STEADY_SAMPLE_STEPS) + 1)
        self._evaluation_snapshot = None
        routed = {weight.expert_parallel_state.num_logical_experts for weight in self.weights}
        redundant = {state.num_redundant_experts_per_rank for state in self._eplb_states}
        assert len(routed) == len(redundant) == 1
        self.num_logical_experts = routed.pop()
        self.num_redundant_experts_per_rank = redundant.pop()
        self.current_placement = torch.stack([state.physical_to_logical for state in self._eplb_states])
        self.in_flight = False
        self.target_placement = None
        self.target_metadata = None
        self.in_flight_started_at = None
        self.evaluation_in_flight = False
        self._evaluation_lock = threading.Lock()
        self._evaluation_result = None
        self._evaluation_error = None
        self._evaluation_thread = None
        self._diagnostics_dir = os.getenv("LIGHTLLM_EPLB_DIAGNOSTICS_DIR") or None
        self._diagnostic_dump_sequence = 0
        self._planner_worker = EPLBPlannerWorker() if self.global_rank == 0 and get_eplb_planner_process() else None
        self._reset_recorded_samples()
        self._set_recording(True)
        # Keep background evaluation collectives separate from the main-thread
        # control/poll collectives: their ordering is intentionally independent.
        self.evaluation_group = dist.new_group(list(range(self.world_size)), backend="gloo")
        self.control_group = dist.new_group(list(range(self.world_size)), backend="gloo")
        self.transfer_group = dist.new_group(list(range(self.world_size)), backend="gloo")
        # This control-group scalar is only touched from the main inference
        # thread, never by the background evaluation thread.
        self._control_ready_count = torch.empty(1, dtype=torch.int32)
        self.transfer = NixlEPLBTransfer(self.weights, self.transfer_group, self.global_rank, self.world_size)
        if self.global_rank == 0:
            logger.info(
                "eplb enabled "
                f"layers={len(self.weights)} num_logical_experts={self.num_logical_experts} "
                f"num_redundant_experts_per_rank={self.num_redundant_experts_per_rank} "
                f"step_interval={self.step_interval} "
                f"rebalance_gain_threshold={self.rebalance_gain_threshold:.4f} "
                f"placement_stickiness={self.placement_stickiness:.4f} "
                f"steady_sample_steps={self.steady_sample_steps} min_rebalance_interval={self.min_rebalance_interval} "
                f"planner_process={self._planner_worker is not None} "
                f"planner_worker={getattr(self._planner_worker, 'info', None)}"
            )

    def before_prefill(self):
        """Clear the next dispatch slots on the forward's own CUDA stream."""
        slots = []
        for state in self._eplb_states:
            capacity = state.route_counter.shape[0]
            index = state.recorded_sample_count % capacity
            # A forward dispatches once, or twice with microbatch overlap.
            # Capacity covers twice the largest forward window, so clearing
            # both next slots also preserves the latest window for one call.
            slots.extend((state.route_counter[index], state.route_counter[(index + 1) % capacity]))
        torch._foreach_zero_(slots)

    def poll(self):
        """Poll only from a globally ordered pre-forward boundary."""
        if self.in_flight:
            self._poll_in_flight()
            return
        if self.evaluation_in_flight and self._evaluation_ready_on_all_ranks():
            self._poll_evaluation()

    def step(self):
        self.prefill_steps += 1
        count = self._eplb_states[0].recorded_sample_count
        previous = self._sample_step_counts[-1]
        if count - previous not in (1, 2) or any(state.recorded_sample_count != count for state in self._eplb_states):
            raise RuntimeError("EPLB forward must record one or two matching dispatches per layer")
        self._sample_step_counts.append(count)
        if self.in_flight or self.evaluation_in_flight or self.prefill_steps < self._next_evaluation_step:
            return
        self._last_evaluation_step = self.prefill_steps
        self._start_evaluation()
        self._next_evaluation_step = (self.prefill_steps // self.sampling_interval + 1) * self.sampling_interval

    def _set_recording(self, enabled: bool):
        for state in self._eplb_states:
            state.recording = enabled

    def _reset_recorded_samples(self):
        counters = [state.route_counter for state in self._eplb_states]
        if counters:
            torch._foreach_zero_(counters)
        for state in self._eplb_states:
            state.recorded_sample_count = 0

    def _control_count(self, value: int) -> torch.Tensor:
        """Return the main-thread-only reusable control collective scalar."""
        return self._control_ready_count.fill_(value)

    def _steady_sample_window_steps(self) -> int:
        # Preserve the original four-step sampling window when the base interval is smaller.
        return min(
            getattr(self, "steady_sample_steps", EPLB_STEADY_SAMPLE_STEPS),
            max(self.step_interval, EPLB_STEADY_SAMPLE_STEPS),
            self.sampling_interval,
        )

    def _prepare_next_sampling_window(self):
        """Schedule the next boundary without resetting the live ring."""
        self._set_recording(True)
        self._next_evaluation_step = (self._last_evaluation_step // self.sampling_interval + 1) * self.sampling_interval

    def _collect_local_samples(self) -> torch.Tensor:
        snapshot = self._evaluation_snapshot
        values = (
            snapshot["count"],
            snapshot["capacity"],
            snapshot["total_count"],
            snapshot["steps"],
            snapshot["completed_step"],
        )
        metadata = torch.tensor([value for item in values for value in (item, -item)], dtype=torch.int64)
        dist.all_reduce(metadata, op=dist.ReduceOp.MIN, group=self.evaluation_group)
        if any(metadata[i] != -metadata[i + 1] for i in range(0, len(metadata), 2)):
            raise RuntimeError("EPLB frozen sample counts differ between ranks")
        boundaries = torch.tensor(
            [value for item in snapshot["forward_end_counts"] for value in (item, -item)], dtype=torch.int64
        )
        dist.all_reduce(boundaries, op=dist.ReduceOp.MIN, group=self.evaluation_group)
        if any(boundaries[i] != -boundaries[i + 1] for i in range(0, len(boundaries), 2)):
            raise RuntimeError("EPLB forward-to-dispatch boundaries differ between ranks")
        samples = snapshot["samples"].cpu()
        # Planning only needs CPU tensors; release the frozen GPU allocation.
        snapshot["samples"] = None
        return samples

    def _commit_layer_metadata(self, layer_index: int):
        eplb_state = self._eplb_states[layer_index]
        logical_to_physical, replica_count = self.target_metadata[layer_index]
        eplb_state.logical_to_physical_map.copy_(logical_to_physical, non_blocking=True)
        eplb_state.logical_replica_count.copy_(replica_count, non_blocking=True)
        placement = self.target_placement[layer_index]
        eplb_state.physical_to_logical = placement.clone()
        eplb_state.placement_generation += 1
        # Keep the CPU state truthful even while only a prefix of layers has
        # been committed; subsequent planning waits for the whole generation.
        self.current_placement[layer_index].copy_(placement)

    def _finish_rebalance(self):
        self.current_placement = self.target_placement
        self.target_placement = None
        self.target_metadata = None
        self.in_flight = False
        self._prepare_next_sampling_window()
        if self.global_rank == 0:
            timing = getattr(self, "_inflight_timing", {})
            logger.info(
                "eplb completed wall_time=%.2fs control_cpu_ms=%.3f control_poll_calls=%s "
                "commit_cpu_ms=%.3f commit_max_ms=%.3f commit_calls=%s "
                "(CPU wall; commit includes existing stream waits, not net stall)",
                time.time() - self.in_flight_started_at,
                timing.get("control_cpu_ms", 0.0),
                timing.get("control_poll_calls", 0),
                timing.get("commit_cpu_ms", 0.0),
                timing.get("commit_max_ms", 0.0),
                timing.get("commit_calls", 0),
            )

    def poll_migration_at_layer_boundary(self):
        """Commit ready layers without publishing a completed generation."""
        if self.in_flight and self.in_flight_layers:
            self._poll_in_flight(finish_generation=False)

    def _poll_in_flight(self, finish_generation=True):
        if not self.in_flight_layers:
            if finish_generation:
                self.transfer.finish()
                self._finish_rebalance()
            return
        local_error = None
        try:
            pending = self.transfer.pending_layers()
        except BaseException as exc:
            pending = []
            local_error = exc
        ready_count = self._control_count(EPLB_CONTROL_ERROR if local_error is not None else len(pending))
        control_started = time.perf_counter()
        dist.all_reduce(ready_count, op=dist.ReduceOp.MIN, group=self.control_group)
        timing = getattr(self, "_inflight_timing", None)
        if timing is not None:
            timing["control_poll_calls"] += 1
            timing["control_cpu_ms"] += (time.perf_counter() - control_started) * 1000.0
        ready_count = int(ready_count.item())
        if ready_count < 0:
            if local_error is not None:
                raise RuntimeError("EPLB transfer worker failed on this rank") from local_error
            raise RuntimeError("EPLB transfer worker failed on another rank")
        if ready_count == 0:
            return
        if ready_count > len(pending) or ready_count > len(self.in_flight_layers):
            raise RuntimeError("EPLB global ready count exceeds the local ordered prefix")
        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        # Previous forward is queued on the shared overlap stream; order the
        # live-weight commit after it. The subsequent wait orders the next forward.
        torch.cuda.current_stream().wait_stream(g_infer_context.get_overlap_stream())
        for layer_index, buffer_index in pending[:ready_count]:
            if layer_index != self.in_flight_layers[0]:
                raise RuntimeError(
                    f"EPLB pending layer {layer_index} does not match expected {self.in_flight_layers[0]}"
                )
            commit_started = time.perf_counter()
            self.transfer.commit(layer_index, buffer_index, lambda: self._commit_layer_metadata(layer_index))
            commit_ms = (time.perf_counter() - commit_started) * 1000.0
            timing = getattr(self, "_inflight_timing", None)
            if timing is not None:
                timing["commit_calls"] += 1
                timing["commit_cpu_ms"] += commit_ms
                timing["commit_max_ms"] = max(timing["commit_max_ms"], commit_ms)
            self.in_flight_layers.pop(0)
        if not self.in_flight_layers and finish_generation:
            self.transfer.finish()
            self._finish_rebalance()

    def _plan_and_broadcast(self, global_load: torch.Tensor):
        """Plan on rank zero and share the serializable result on the evaluation group."""
        replica_hash_load = global_load[..., 1:] if global_load.ndim == 5 else None
        if replica_hash_load is not None:
            global_load = global_load[..., 0]
        result = None
        local_error = None
        current_placement = self._evaluation_snapshot["placement"]
        if self.global_rank == 0:
            try:
                minimum = self.num_logical_experts * EPLB_MIN_AVG_TOKENS_PER_EXPERT
                layer_samples = global_load.sum(dim=(0, 2, 3))
                if torch.any(layer_samples < minimum):
                    result = {
                        "kind": "insufficient",
                        "minimum_layer_samples": int(layer_samples.min().item()),
                        "minimum": minimum,
                    }
                else:
                    settings = {
                        "node_world_size": self.node_world_size,
                        "expert_alignment": EPLB_EXPERT_ALIGNMENT,
                        "stickiness": self.placement_stickiness,
                    }
                    if replica_hash_load is not None:
                        settings["replica_hash_load"] = replica_hash_load
                    if self._initial_plan_complete:
                        settings["incremental"] = True
                    worker = getattr(self, "_planner_worker", None)
                    if worker is None:
                        candidate = plan_eplb_candidate(global_load, current_placement, **settings)
                        candidate_timing = {"compute_ms": None, "roundtrip_ms": None}
                    else:
                        candidate, candidate_timing = worker.plan(
                            global_load.cpu(), current_placement.cpu(), **settings
                        )
                    select_align_started = time.perf_counter()
                    placement, improved, metrics, before_load, after_load = select_improving_placements(
                        global_load,
                        current_placement,
                        candidate,
                        expert_alignment=EPLB_EXPERT_ALIGNMENT,
                        node_world_size=self.node_world_size,
                        rebalance_gain_threshold=self.rebalance_gain_threshold,
                        replica_hash_load=replica_hash_load,
                    )
                    if bool(torch.any(improved)):
                        # A planner placement identifies experts by rank, not
                        # by redundant slot. Canonicalize every selected row
                        # before broadcasting so transfer, metadata, and the
                        # next current_placement all describe the same live
                        # physical expert rows.
                        placement = placement.clone()
                        for layer_index in torch.nonzero(improved, as_tuple=False).flatten().tolist():
                            placement[layer_index] = align_target_placement(
                                current_placement[layer_index], placement[layer_index]
                            )
                    select_align_ms = (time.perf_counter() - select_align_started) * 1000.0
                    result = {
                        "candidate_timing_ms": {
                            **candidate_timing,
                            "select_align_ms": select_align_ms,
                        },
                        "kind": "planned" if bool(torch.any(improved)) else "no_improvement",
                        "placement": placement,
                        "improved": improved,
                        "before": _imbalance_summary(before_load),
                        "after": _imbalance_summary(after_load),
                        **metrics,
                    }
            except BaseException as exc:
                local_error = exc
                result = {"kind": "error", "message": f"{type(exc).__name__}: {exc}"}
        broadcast_started = time.perf_counter()
        if self.world_size > 1:
            result_list = [result]
            dist.broadcast_object_list(result_list, src=0, group=self.evaluation_group)
            result = result_list[0]
        broadcast_ms = (time.perf_counter() - broadcast_started) * 1000.0
        if result["kind"] == "error":
            if local_error is not None:
                raise RuntimeError("EPLB planner failed on rank zero") from local_error
            raise RuntimeError(f"EPLB planner failed on rank zero: {result['message']}")
        result["broadcast_ms"] = broadcast_ms
        return result

    def _dump_evaluation_samples(self, global_load, recorded_sample_count, sample_window_steps):
        diagnostics_dir = getattr(self, "_diagnostics_dir", None)
        if not diagnostics_dir or self.global_rank != 0:
            return 0.0
        started = time.perf_counter()
        try:
            directory = Path(diagnostics_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self._diagnostic_dump_sequence = getattr(self, "_diagnostic_dump_sequence", 0) + 1
            payload = {
                "global_load": global_load[..., 0] if global_load.ndim == 5 else global_load,
                "current_placement": self._evaluation_snapshot["placement"],
                "prefill_steps": self._evaluation_snapshot["completed_step"],
                "recorded_sample_count": recorded_sample_count,
                "sample_window_steps": sample_window_steps,
            }
            torch.save(payload, directory / f"evaluation-{self._diagnostic_dump_sequence:06d}.pt")
        except Exception as exc:
            logger.warning("EPLB diagnostics dump disabled after write failure: %s", exc)
            self._diagnostics_dir = None
        return (time.perf_counter() - started) * 1000.0

    def _evaluate_after_event(self, event: torch.cuda.Event):
        """Run the CPU/Gloo planning phase after the frozen CUDA counters are ready."""
        try:
            timing_started = time.perf_counter()
            thread_cpu_started = time.thread_time()
            torch.cuda.set_device(self._eplb_states[0].route_counter.device)
            event.synchronize()
            event_wait_ms = (time.perf_counter() - timing_started) * 1000.0
            collect_started = time.perf_counter()
            local_load = self._collect_local_samples()
            recorded_sample_count = int(local_load.shape[0])
            sample_window_steps = self._evaluation_snapshot["steps"]
            num_nodes = self.world_size // self.node_world_size
            # Preserve source nodes until physical-replica loads are combined;
            # DeepEP applies expert alignment after traffic from all sources
            # reaches each destination expert.
            global_load = torch.zeros((*local_load.shape[:2], num_nodes, *local_load.shape[2:]), dtype=local_load.dtype)
            global_load[:, :, self.global_rank // self.node_world_size] = local_load
            dist.all_reduce(global_load, op=dist.ReduceOp.SUM, group=self.evaluation_group)
            collect_reduce_ms = (time.perf_counter() - collect_started) * 1000.0
            diagnostic_dump_ms = self._dump_evaluation_samples(global_load, recorded_sample_count, sample_window_steps)
            plan_started = time.perf_counter()
            result = self._plan_and_broadcast(global_load)
            plan_broadcast_ms = (time.perf_counter() - plan_started) * 1000.0
            prepare_started = time.perf_counter()
            result["recorded_sample_count"] = recorded_sample_count
            result["sample_window_steps"] = sample_window_steps
            if result["kind"] == "planned":
                metadata = [None] * len(self.weights)
                layer_plans = []
                improved_layer_indices = torch.nonzero(result["improved"], as_tuple=False).flatten()
                if improved_layer_indices.numel():
                    maps_for_improved_layers, counts_for_improved_layers = build_logical_to_physical_maps_for_layers(
                        result["placement"][improved_layer_indices],
                        self.num_logical_experts,
                        source_rank=self.global_rank,
                        node_world_size=self.node_world_size,
                        replica_rotation_rank=self._eplb_states[0].replica_rotation_rank,
                    )
                    for improved_layer_offset, layer_index in enumerate(improved_layer_indices.tolist()):
                        placement = result["placement"][layer_index]
                        metadata[layer_index] = (
                            maps_for_improved_layers[improved_layer_offset],
                            counts_for_improved_layers[improved_layer_offset],
                        )
                        layer_plans.append(
                            (
                                layer_index,
                                build_transfer_plan(
                                    self.current_placement[layer_index],
                                    placement,
                                    self.num_logical_experts,
                                    self.world_size,
                                    self.node_world_size,
                                ),
                            )
                        )
                result["metadata"] = metadata
                result["layer_plans"] = layer_plans
                result["prepared_batches"] = self.transfer.prepare_transfer(layer_plans)
            prepare_ms = (time.perf_counter() - prepare_started) * 1000.0
            candidate_timing = result.get("candidate_timing_ms", {})
            result["evaluation_timing_ms"] = {
                "event_wait": event_wait_ms,
                "collect_reduce": collect_reduce_ms,
                "plan_broadcast": plan_broadcast_ms,
                "candidate_compute": candidate_timing.get("compute_ms"),
                "candidate_roundtrip": candidate_timing.get("roundtrip_ms"),
                "select_align": candidate_timing.get("select_align_ms"),
                "broadcast": result.get("broadcast_ms"),
                "prepare": prepare_ms,
                "diagnostic_dump": diagnostic_dump_ms,
                "total": (time.perf_counter() - timing_started) * 1000.0,
                "parent_thread_cpu": (time.thread_time() - thread_cpu_started) * 1000.0,
            }
            with self._evaluation_lock:
                self._evaluation_result = result
        except BaseException as exc:
            with self._evaluation_lock:
                self._evaluation_error = exc

    def _start_evaluation(self):
        with self._evaluation_lock:
            self._evaluation_result = None
            self._evaluation_error = None
        self._freeze_evaluation_samples()
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self.evaluation_in_flight = True
        self._evaluation_thread = threading.Thread(target=self._evaluate_after_event, args=(event,), daemon=True)
        self._evaluation_thread.start()

    def _freeze_evaluation_samples(self):
        """Freeze a chronological forward window before live-ring writes resume."""
        requested = self.step_interval if self._use_full_window else self._steady_sample_window_steps()
        steps = min(requested, len(self._sample_step_counts) - 1)
        total = self._sample_step_counts[-1]
        count = total - self._sample_step_counts[-steps - 1]
        counters = [state.route_counter for state in self._eplb_states]
        capacity = counters[0].shape[0]
        if (
            steps == 0
            or count > capacity
            or any(
                state.recorded_sample_count != total or state.route_counter.shape != counters[0].shape
                for state in self._eplb_states
            )
        ):
            raise RuntimeError("EPLB continuous window exceeds its ring capacity or differs between layers")
        indices = torch.arange(total - count, total, dtype=torch.int64, device=counters[0].device) % capacity
        samples = torch.stack(counters, dim=1).index_select(0, indices)
        self._evaluation_snapshot = dict(
            samples=samples,
            count=count,
            total_count=total,
            capacity=capacity,
            steps=steps,
            placement=self.current_placement.clone(),
            completed_step=self.prefill_steps,
            forward_end_counts=tuple(self._sample_step_counts)[-steps - 1 :],
        )

    def _poll_evaluation(self):
        if not self.evaluation_in_flight:
            return False
        with self._evaluation_lock:
            error = self._evaluation_error
            result = self._evaluation_result
            if error is not None or result is not None:
                self._evaluation_result = None
                self._evaluation_error = None
        if error is not None:
            self._evaluation_thread.join()
            self.evaluation_in_flight = False
            self._evaluation_thread = None
            raise error
        if result is None:
            return True
        self._evaluation_thread.join()
        self.evaluation_in_flight = False
        self._evaluation_thread = None
        if self.global_rank == 0 and result.get("evaluation_timing_ms"):
            logger.info(
                "eplb evaluation CPU wall ms event_wait=%.3f collect_reduce=%.3f "
                "plan_broadcast=%.3f prepare=%.3f diagnostic_dump=%.3f total=%.3f "
                "candidate_compute=%s candidate_roundtrip=%s select_align=%s broadcast=%s parent_thread_cpu=%s",
                *[
                    result["evaluation_timing_ms"][key]
                    for key in (
                        "event_wait",
                        "collect_reduce",
                        "plan_broadcast",
                        "prepare",
                        "diagnostic_dump",
                        "total",
                        "candidate_compute",
                        "candidate_roundtrip",
                        "select_align",
                        "broadcast",
                        "parent_thread_cpu",
                    )
                ],
            )
        if result["kind"] == "insufficient":
            self._use_full_window = True
            self._prepare_next_sampling_window()
            return False
        if result["kind"] == "no_improvement":
            self._initial_plan_complete = True
            self._use_full_window = False
            self.sampling_interval = max(self.step_interval, self.min_rebalance_interval)
            if self.global_rank == 0:
                logger.info(
                    "eplb skip rearrangement: no model improvement model_imbalance_ratio=%.4f "
                    "candidate_model_imbalance_ratio=%.4f candidate_rebalance_gain=%.4f "
                    "candidate_changed_layer_count=%s actual_changed_layer_count=0 next_sampling_interval=%s "
                    "recorded_sample_count=%s sample_window_steps=%s",
                    result["model_imbalance_ratio"],
                    result["candidate_model_imbalance_ratio"],
                    result["candidate_rebalance_gain"],
                    result["candidate_changed_layer_count"],
                    self.sampling_interval,
                    result.get("recorded_sample_count"),
                    result.get("sample_window_steps"),
                )
            self._prepare_next_sampling_window()
            return False
        self._start_rebalance(result)
        return True

    def _evaluation_ready_on_all_ranks(self) -> bool:
        with self._evaluation_lock:
            local_error = self._evaluation_error
            local_result = self._evaluation_result
        local_status = EPLB_CONTROL_ERROR if local_error is not None else int(local_result is not None)
        ready_count = self._control_count(local_status)
        dist.all_reduce(ready_count, op=dist.ReduceOp.MIN, group=self.control_group)
        ready_count = int(ready_count.item())
        if ready_count < 0:
            if local_error is not None:
                raise RuntimeError("EPLB evaluation failed on this rank") from local_error
            raise RuntimeError("EPLB evaluation failed on another rank")
        return bool(ready_count)

    def _start_rebalance(self, result):
        placement = result["placement"]
        layer_plans = result["layer_plans"]
        self.sampling_interval = max(self.step_interval, getattr(self, "min_rebalance_interval", 0))
        self._initial_plan_complete = True
        self._use_full_window = False
        self.target_placement = placement
        self.target_metadata = result["metadata"]
        self.in_flight_layers = [layer_index for layer_index, _ in layer_plans]
        self.in_flight = True
        self.in_flight_started_at = time.time()
        self._inflight_timing = {
            "control_poll_calls": 0,
            "control_cpu_ms": 0.0,
            "commit_calls": 0,
            "commit_cpu_ms": 0.0,
            "commit_max_ms": 0.0,
        }
        self.transfer.start(layer_plans, result["prepared_batches"])
        if self.global_rank == 0:
            actual_changed_slot_count = sum(len(plan) for _, plan in layer_plans)
            cross_node_transfer_count = sum(
                step.src_rank // self.node_world_size != step.dst_rank // self.node_world_size
                for _, plan in layer_plans
                for step in plan
            )
            logger.info(
                "eplb started prefill_steps=%s max_before=%.4f max_after=%.4f p95_before=%.4f p95_after=%.4f "
                "model_imbalance_ratio=%.4f candidate_model_imbalance_ratio=%.4f "
                "candidate_rebalance_gain=%.4f candidate_changed_layer_count=%s "
                "actual_changed_layer_count=%s actual_changed_slot_count=%s cross_node_transfer_count=%s "
                "recorded_sample_count=%s sample_window_steps=%s",
                self.prefill_steps,
                result["before"]["max"],
                result["after"]["max"],
                result["before"]["p95"],
                result["after"]["p95"],
                result["model_imbalance_ratio"],
                result["candidate_model_imbalance_ratio"],
                result["candidate_rebalance_gain"],
                result["candidate_changed_layer_count"],
                len(layer_plans),
                actual_changed_slot_count,
                cross_node_transfer_count,
                result.get("recorded_sample_count"),
                result.get("sample_window_steps"),
            )


def _imbalance_summary(rank_load: torch.Tensor) -> Dict[str, float]:
    if rank_load.ndim == 2:
        critical = rank_load.max(dim=1).values
        mean = rank_load.mean(dim=1)
    elif rank_load.ndim == 3:
        critical = rank_load.max(dim=2).values.sum(dim=0)
        mean = rank_load.mean(dim=2).sum(dim=0)
    else:
        raise ValueError("rank_load must be [layers, ranks] or [samples, layers, ranks]")
    layer_imbalance = critical / mean.clamp_min(1.0)
    sorted_imbalance = torch.sort(layer_imbalance).values
    p95_index = max(0, (95 * layer_imbalance.numel() + 99) // 100 - 1)
    return {
        "max": float(layer_imbalance.max().item()),
        "p95": float(sorted_imbalance[p95_index].item()),
    }
