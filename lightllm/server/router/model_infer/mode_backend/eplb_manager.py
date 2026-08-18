import threading
import time
from typing import Dict, Optional

import torch
import torch.distributed as dist

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_placement import (
    build_initial_redundant_expert_ids,
    build_logical_to_physical_map,
    plan_redundant_experts,
    _select_improving_placements_with_loads,
)
from lightllm.server.router.model_infer.mode_backend.eplb_transfer import (
    NixlEPLBTransfer,
    build_transfer_plan,
)
from lightllm.utils.dist_utils import get_global_rank, get_global_world_size, get_node_world_size
from lightllm.utils.envs_utils import (
    get_eplb_rebalance_gain_threshold,
    get_prefill_eplb_step_interval,
)
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)
EPLB_MIN_AVG_TOKENS_PER_EXPERT = 100
EPLB_EXPERT_ALIGNMENT = 128


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


def _find_fused_moe_weights(model):
    weights_by_id = {}
    for layer in model.trans_layers_weight:
        for value in getattr(layer, "__dict__", {}).values():
            if isinstance(value, FusedMoeWeight) and value.enable_ep_moe:
                weights_by_id[id(value)] = value
    return sorted(weights_by_id.values(), key=lambda weight: weight.layer_num_)


class EPLBManager:
    """Online EPLB with asynchronous GPU expert migration."""

    def __init__(self, model: TpPartBaseModel):
        self.weights = _find_fused_moe_weights(model)
        assert self.weights, "EPLB requires at least one EP MoE layer"
        self.global_rank = get_global_rank()
        self.world_size = get_global_world_size()
        self.node_world_size = get_node_world_size()
        self._eplb_states = [weight.expert_parallel_state.eplb for weight in self.weights]
        self.step_interval = get_prefill_eplb_step_interval()
        self.rebalance_gain_threshold = get_eplb_rebalance_gain_threshold()
        self.sampling_interval = self.step_interval
        self.prefill_steps = 0
        self.rebalanced = False
        routed = {weight.expert_parallel_state.logical_experts for weight in self.weights}
        redundant = {state.redundant_experts_per_rank for state in self._eplb_states}
        assert len(routed) == len(redundant) == 1
        self.num_logical_experts = routed.pop()
        self.redundant_experts_per_rank = redundant.pop()
        initial = build_initial_redundant_expert_ids(
            self.num_logical_experts, self.world_size, self.redundant_experts_per_rank
        )
        self.current_placement = initial.unsqueeze(0).expand(len(self.weights), -1, -1).clone()
        self.in_flight = False
        self.target_placement = None
        self.target_metadata = None
        self.in_flight_started_at = None
        self.evaluation_in_flight = False
        self._evaluation_lock = threading.Lock()
        self._evaluation_result = None
        self._evaluation_error = None
        self._evaluation_thread = None
        # A fresh manager starts with one continuous base window. After a
        # sufficient evaluation, steady state returns to the cheap sparse
        # probe. An insufficient sparse probe schedules one fresh continuous
        # base window before the next fixed sampling boundary.
        self._continuous_collection_start_step: Optional[int] = None
        self._continuous_collection_end_step: Optional[int] = self.step_interval
        self._sampling_pending = False
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
                f"layers={len(self.weights)} logical_experts={self.num_logical_experts} "
                f"redundant_experts_per_rank={self.redundant_experts_per_rank} "
                f"step_interval={self.step_interval} "
                f"rebalance_gain_threshold={self.rebalance_gain_threshold:.4f}"
            )

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
        tensor = getattr(self, "_control_ready_count", None)
        if tensor is None:
            tensor = self._control_ready_count = torch.empty(1, dtype=torch.int32)
        return tensor.fill_(value)

    def _clear_continuous_collection(self):
        self._continuous_collection_start_step = None
        self._continuous_collection_end_step = None

    def _begin_continuous_collection(self):
        minimum_end = self.prefill_steps + self.step_interval
        collection_end = -(-minimum_end // self.sampling_interval) * self.sampling_interval
        self._reset_recorded_samples()
        self._sampling_pending = False
        self._continuous_collection_start_step = collection_end - self.step_interval
        self._continuous_collection_end_step = collection_end
        self._set_recording(self._continuous_collection_start_step == self.prefill_steps)

    def _prepare_next_sampling_window(self):
        """Clear the current window and arm the next sparse sampling window."""
        self._clear_continuous_collection()
        self._reset_recorded_samples()
        self._sampling_pending = False
        self._set_recording(self.sampling_interval == 1)

    @staticmethod
    def _recent_ring_samples(counter: torch.Tensor, recorded_sample_count: int) -> torch.Tensor:
        """Return the newest ring rows in chronological order."""
        capacity = counter.shape[0]
        available = min(recorded_sample_count, capacity)
        if available == 0:
            return counter[:0]
        start = (recorded_sample_count - available) % capacity
        indices = (torch.arange(available, dtype=torch.int64, device=counter.device) + start) % capacity
        return counter.index_select(0, indices)

    def _collect_local_samples(self) -> torch.Tensor:
        counters = [state.route_counter for state in self._eplb_states]
        capacities = [counter.shape[0] for counter in counters]
        if len(set(capacities)) != 1 or any(counter.ndim != 2 for counter in counters):
            raise RuntimeError("EPLB sample capacities differ between layers")
        counts = [state.recorded_sample_count for state in self._eplb_states]
        if len(set(counts)) != 1:
            raise RuntimeError("EPLB recorded sample counts differ between layers")
        sample_count = counts[0]
        # Validate the metadata before copying the newest rows to the CPU.
        metadata = torch.tensor([sample_count, -sample_count, capacities[0], -capacities[0]], dtype=torch.int64)
        dist.all_reduce(metadata, op=dist.ReduceOp.MIN, group=self.evaluation_group)
        if metadata[0] != -metadata[1] or metadata[2] != -metadata[3]:
            raise RuntimeError("EPLB recorded sample count or capacity differs between ranks")
        # Stack the fixed-size ring buffers in one GPU launch.  Slicing each
        # layer before stacking turns a single launch into one index_select per
        # MoE layer and is measurably slower in the normal sparse path.
        counter_samples = torch.stack(counters, dim=1)
        return self._recent_ring_samples(counter_samples, sample_count).cpu()

    def _commit_layer_metadata(self, layer_index: int):
        eplb_state = self._eplb_states[layer_index]
        metadata = self.target_metadata[layer_index] if self.target_metadata is not None else None
        if metadata is None:
            # Unit tests and external callers may still construct a legacy
            # result by hand. Production planned results always precompute.
            metadata = build_logical_to_physical_map(
                self.target_placement[layer_index],
                self.num_logical_experts,
                source_rank=self.global_rank,
                node_world_size=self.node_world_size,
            )
        logical_to_physical, replica_count = metadata
        eplb_state.logical_to_physical_map.copy_(logical_to_physical, non_blocking=True)
        eplb_state.logical_replica_count.copy_(replica_count, non_blocking=True)

    def _finish_rebalance(self):
        self.current_placement = self.target_placement
        self.target_placement = None
        self.target_metadata = None
        self.in_flight = False
        self.rebalanced = True
        self._prepare_next_sampling_window()
        if self.global_rank == 0:
            logger.info(f"eplb completed wall_time={time.time() - self.in_flight_started_at:.2f}s")

    def _poll_in_flight(self):
        pending = self.transfer.pending_layers()  # Propagates worker errors on the main thread.
        ready_count = self._control_count(len(pending))
        dist.all_reduce(ready_count, op=dist.ReduceOp.MIN, group=self.control_group)
        ready_count = int(ready_count.item())
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
            self.transfer.commit(layer_index, buffer_index, lambda: self._commit_layer_metadata(layer_index))
            self.in_flight_layers.pop(0)
        if not self.in_flight_layers:
            self.transfer.finish()
            self._finish_rebalance()

    def _plan_and_broadcast(self, global_load: torch.Tensor):
        """Plan on rank zero and share the serializable result on the evaluation group."""
        result = None
        if self.global_rank == 0:
            minimum = self.num_logical_experts * EPLB_MIN_AVG_TOKENS_PER_EXPERT
            layer_samples = global_load.sum(dim=(0, 2, 3))
            if torch.any(layer_samples < minimum):
                result = {
                    "kind": "insufficient",
                    "minimum_layer_samples": int(layer_samples.min().item()),
                    "minimum": minimum,
                }
            else:
                candidate = plan_redundant_experts(
                    global_load,
                    self.world_size,
                    self.redundant_experts_per_rank,
                    expert_alignment=EPLB_EXPERT_ALIGNMENT,
                    node_world_size=self.node_world_size,
                )
                placement, improved, metrics, before_load, after_load = _select_improving_placements_with_loads(
                    global_load,
                    self.current_placement,
                    candidate,
                    expert_alignment=EPLB_EXPERT_ALIGNMENT,
                    node_world_size=self.node_world_size,
                    rebalance_gain_threshold=self.rebalance_gain_threshold,
                )
                result = {
                    "kind": "planned" if bool(torch.any(improved)) else "no_improvement",
                    "placement": placement,
                    "improved": improved,
                    "before": _imbalance_summary(before_load),
                    "after": _imbalance_summary(after_load),
                    **metrics,
                }
        if self.world_size > 1:
            result_list = [result]
            dist.broadcast_object_list(result_list, src=0, group=self.evaluation_group)
            result = result_list[0]
        return result

    def _evaluate_after_event(self, event: torch.cuda.Event):
        """Run the CPU/Gloo planning phase after the frozen CUDA counters are ready."""
        try:
            torch.cuda.set_device(self._eplb_states[0].route_counter.device)
            event.synchronize()
            local_load = self._collect_local_samples()
            num_nodes = self.world_size // self.node_world_size
            # Preserve source nodes until physical-replica loads are combined;
            # DeepEP applies expert alignment after traffic from all sources
            # reaches each destination expert.
            global_load = torch.zeros((*local_load.shape[:2], num_nodes, local_load.shape[2]), dtype=local_load.dtype)
            global_load[:, :, self.global_rank // self.node_world_size] = local_load
            dist.all_reduce(global_load, op=dist.ReduceOp.SUM, group=self.evaluation_group)
            result = self._plan_and_broadcast(global_load)
            if result["kind"] == "planned":
                metadata = [None] * len(self.weights)
                layer_plans = []
                for layer_index, is_improved in enumerate(result["improved"]):
                    if bool(is_improved):
                        placement = result["placement"][layer_index]
                        metadata[layer_index] = build_logical_to_physical_map(
                            placement,
                            self.num_logical_experts,
                            source_rank=self.global_rank,
                            node_world_size=self.node_world_size,
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
            with self._evaluation_lock:
                self._evaluation_result = result
        except BaseException as exc:
            with self._evaluation_lock:
                self._evaluation_error = exc

    def _start_evaluation(self):
        with self._evaluation_lock:
            self._evaluation_result = None
            self._evaluation_error = None
        self._set_recording(False)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self.evaluation_in_flight = True
        self._evaluation_thread = threading.Thread(target=self._evaluate_after_event, args=(event,), daemon=True)
        self._evaluation_thread.start()

    def _poll_evaluation(self):
        if not getattr(self, "evaluation_in_flight", False):
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
        if result["kind"] == "insufficient":
            from_continuous_window = self._continuous_collection_end_step is not None
            if from_continuous_window:
                self.sampling_interval = min(self.sampling_interval * 4, self.step_interval * 16)
                self._prepare_next_sampling_window()
            else:
                self._begin_continuous_collection()
            if self.global_rank == 0:
                if from_continuous_window:
                    logger.info(
                        "eplb insufficient samples: prefill_steps=%s minimum_layer_samples=%s required=%s "
                        "next_sampling_interval=%s",
                        self.prefill_steps,
                        result["minimum_layer_samples"],
                        result["minimum"],
                        self.sampling_interval,
                    )
                else:
                    logger.info(
                        "eplb insufficient samples: prefill_steps=%s minimum_layer_samples=%s required=%s "
                        "scheduled_fresh_window_start=%s scheduled_fresh_window_end=%s",
                        self.prefill_steps,
                        result["minimum_layer_samples"],
                        result["minimum"],
                        self._continuous_collection_start_step,
                        self._continuous_collection_end_step,
                    )
            return False
        if result["kind"] == "no_improvement":
            self.sampling_interval = min(self.sampling_interval * 4, self.step_interval * 16)
            if self.global_rank == 0:
                logger.info(
                    "eplb skip rearrangement: no model improvement model_imbalance_ratio=%.4f "
                    "candidate_model_imbalance_ratio=%.4f candidate_rebalance_gain=%.4f "
                    "candidate_changed_layer_count=%s actual_changed_layer_count=0 next_sampling_interval=%s",
                    result["model_imbalance_ratio"],
                    result["candidate_model_imbalance_ratio"],
                    result["candidate_rebalance_gain"],
                    result["candidate_changed_layer_count"],
                    self.sampling_interval,
                )
            self._prepare_next_sampling_window()
            return False
        self._start_rebalance(result)
        return True

    def _evaluation_ready_on_all_ranks(self) -> bool:
        with self._evaluation_lock:
            local_ready = self._evaluation_error is not None or self._evaluation_result is not None
        ready_count = self._control_count(int(local_ready))
        dist.all_reduce(ready_count, op=dist.ReduceOp.MIN, group=self.control_group)
        return bool(ready_count.item())

    def _start_rebalance(self, result):
        placement, improved = result["placement"], result["improved"]
        layer_plans = result.get("layer_plans")
        if layer_plans is None:
            layer_plans = []
            for layer_index, is_improved in enumerate(improved):
                if bool(is_improved):
                    plan = build_transfer_plan(
                        self.current_placement[layer_index],
                        placement[layer_index],
                        self.num_logical_experts,
                        self.world_size,
                        self.node_world_size,
                    )
                    layer_plans.append((layer_index, plan))
        self.sampling_interval = self.step_interval
        self._clear_continuous_collection()
        self._reset_recorded_samples()
        self.target_placement = placement
        self.target_metadata = result.get("metadata")
        self.in_flight_layers = [layer_index for layer_index, _ in layer_plans]
        self.in_flight = True
        self.in_flight_started_at = time.time()
        self.transfer.start(layer_plans)
        if self.global_rank == 0:
            logger.info(
                "eplb started prefill_steps=%s max_before=%.4f max_after=%.4f p95_before=%.4f p95_after=%.4f "
                "model_imbalance_ratio=%.4f candidate_model_imbalance_ratio=%.4f "
                "candidate_rebalance_gain=%.4f candidate_changed_layer_count=%s "
                "actual_changed_layer_count=%s",
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
            )

    def poll(self):
        """Poll only from a globally ordered pre-forward boundary."""
        if self.in_flight:
            self._poll_in_flight()
            return
        if self.evaluation_in_flight and self._evaluation_ready_on_all_ranks():
            self._poll_evaluation()

    def step(self):
        if self.in_flight or self.evaluation_in_flight:
            return
        self.prefill_steps += 1
        continuous_start = self._continuous_collection_start_step
        continuous_end = self._continuous_collection_end_step
        if continuous_end is not None:
            if continuous_start is not None and self.prefill_steps == continuous_start:
                self._set_recording(True)
            if self.prefill_steps >= continuous_end:
                self._start_evaluation()
            return
        sampling_interval = self.sampling_interval
        phase = self.prefill_steps % sampling_interval
        if self._sampling_pending:
            self._sampling_pending = False
            self._start_evaluation()
            return
        if sampling_interval == 1:
            self._start_evaluation()
            return
        if phase == sampling_interval - 1:
            self._reset_recorded_samples()
            self._sampling_pending = True
            self._set_recording(True)
