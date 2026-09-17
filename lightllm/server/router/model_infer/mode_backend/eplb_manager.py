import threading
import time
from typing import Dict, Optional

import torch
import torch.distributed as dist

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import (
    FusedMoeWeight,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_placement import (
    build_initial_local_expert_ids,
    build_logical_to_physical_maps_for_layers,
    plan_redundant_experts,
    select_improving_placements,
)
from lightllm.server.router.model_infer.mode_backend.eplb_transfer import (
    PinnedMemoryEPLBTransfer,
    align_target_placement,
    build_transfer_plan,
)
from lightllm.server.metrics.manager import MetricClient
from lightllm.utils.dist_utils import (
    get_global_rank,
    get_global_world_size,
    get_node_world_size,
)
from lightllm.utils.envs_utils import (
    get_eplb_placement_stickiness,
    get_eplb_rebalance_gain_threshold,
    get_prefill_eplb_step_interval,
)
from lightllm.utils.log_utils import init_logger
from lightllm.utils.shm_port_args import get_shm_port_args

logger = init_logger(__name__)
EPLB_MIN_AVG_TOKENS_PER_EXPERT = 100
EPLB_EXPERT_ALIGNMENT = 128
EPLB_CONTROL_ERROR = -1
EPLB_STEADY_SAMPLE_STEPS = 4
EPLB_EXPERT_IMBALANCE_RATIO_METRIC = "lightllm_eplb_topk_expert_imbalance_ratio"


class EPLBManager:
    """Online EPLB with asynchronous GPU expert migration."""

    def __init__(self, model: TpPartBaseModel):
        self.weights = _find_fused_moe_weights(model)
        assert self.weights, "EPLB requires at least one EP MoE layer"
        self.global_rank = get_global_rank()
        self.world_size = get_global_world_size()
        self.node_world_size = get_node_world_size()
        self._eplb_impls = [weight.fuse_moe_impl for weight in self.weights]
        self.step_interval = get_prefill_eplb_step_interval()
        self.rebalance_gain_threshold = get_eplb_rebalance_gain_threshold()
        self.placement_stickiness = get_eplb_placement_stickiness()
        self.sampling_interval = self.step_interval
        self.prefill_steps = 0
        routed = {weight.fuse_moe_impl.n_routed_experts for weight in self.weights}
        redundant = {impl.num_redundant_experts_per_rank for impl in self._eplb_impls}
        assert len(routed) == len(redundant) == 1
        self.num_logical_experts = routed.pop()
        self.num_redundant_experts_per_rank = redundant.pop()
        num_primary_experts_per_rank = self.num_logical_experts // self.world_size
        initial_local_expert_ids_by_rank = build_initial_local_expert_ids(
            self.num_logical_experts,
            self.world_size,
            self.num_redundant_experts_per_rank,
        )
        initial_redundant_expert_ids_by_rank = [
            expert_ids[num_primary_experts_per_rank:] for expert_ids in initial_local_expert_ids_by_rank
        ]
        self.current_placement = torch.tensor(
            [initial_redundant_expert_ids_by_rank for _ in self.weights],
            dtype=torch.int64,
        )
        self.in_flight = False
        self.target_placement = None
        self.target_metadata = None
        self.in_flight_started_at = None
        self.evaluation_in_flight = False
        self._evaluation_lock = threading.Lock()
        self._evaluation_result = None
        self._evaluation_error = None
        self._evaluation_thread = None
        self.metric_client = None
        # A fresh manager starts with one continuous base window. After a
        # sufficient evaluation, steady state returns to the cheap sparse
        # probe. An insufficient sparse probe schedules one fresh continuous
        # base window before the next fixed sampling boundary.
        self._continuous_collection_start_step: Optional[int] = None
        self._continuous_collection_end_step: Optional[int] = self.step_interval
        self._steady_collection_end_step: Optional[int] = None
        self._reset_route_counters()
        self._set_recording(True)
        # Keep background evaluation collectives separate from the main-thread
        # control/poll collectives: their ordering is intentionally independent.
        self.evaluation_group = dist.new_group(list(range(self.world_size)), backend="gloo")
        self.control_group = dist.new_group(list(range(self.world_size)), backend="gloo")
        self.transfer_group = dist.new_group(list(range(self.world_size)), backend="gloo")
        # This control-group scalar is only touched from the main inference
        # thread, never by the background evaluation thread.
        self._control_ready_count = torch.empty(1, dtype=torch.int32)
        self.transfer = PinnedMemoryEPLBTransfer(self.weights, self.transfer_group, self.global_rank, self.world_size)
        if self.global_rank == 0:
            logger.info(
                "eplb enabled "
                f"layers={len(self.weights)} num_logical_experts={self.num_logical_experts} "
                f"num_redundant_experts_per_rank={self.num_redundant_experts_per_rank} "
                f"step_interval={self.step_interval} "
                f"rebalance_gain_threshold={self.rebalance_gain_threshold:.4f} "
                f"placement_stickiness={self.placement_stickiness:.4f}"
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
        steady_collection_end_step = self._steady_collection_end_step
        if steady_collection_end_step is not None:
            if self.prefill_steps >= steady_collection_end_step:
                self._steady_collection_end_step = None
                self._start_evaluation()
            return
        if sampling_interval == 1:
            self._start_evaluation()
            return
        if phase == sampling_interval - self._steady_sample_window_steps():
            self._arm_steady_collection(self.prefill_steps + self._steady_sample_window_steps())

    def _set_recording(self, enabled: bool):
        for impl in self._eplb_impls:
            impl.recording = enabled

    def _reset_route_counters(self):
        counters = [impl.route_counter for impl in self._eplb_impls]
        if counters:
            torch._foreach_zero_(counters)

    def _control_count(self, value: int) -> torch.Tensor:
        """Return the main-thread-only reusable control collective scalar."""
        return self._control_ready_count.fill_(value)

    def _clear_continuous_collection(self):
        self._continuous_collection_start_step = None
        self._continuous_collection_end_step = None

    def _steady_sample_window_steps(self) -> int:
        return min(EPLB_STEADY_SAMPLE_STEPS, self.sampling_interval)

    def _arm_steady_collection(self, collection_end_step: int):
        """Start the fixed sparse window without moving its evaluation boundary."""
        self._reset_route_counters()
        self._steady_collection_end_step = collection_end_step
        self._set_recording(True)

    def _begin_continuous_collection(self):
        minimum_end = self.prefill_steps + self.step_interval
        collection_end = -(-minimum_end // self.sampling_interval) * self.sampling_interval
        self._reset_route_counters()
        self._steady_collection_end_step = None
        self._continuous_collection_start_step = collection_end - self.step_interval
        self._continuous_collection_end_step = collection_end
        self._set_recording(self._continuous_collection_start_step == self.prefill_steps)

    def _prepare_next_sampling_window(self):
        """Clear the current window and arm the next sparse sampling window."""
        self._clear_continuous_collection()
        self._steady_collection_end_step = None
        if self.sampling_interval == 1:
            self._reset_route_counters()
            self._set_recording(True)
        elif self.sampling_interval <= EPLB_STEADY_SAMPLE_STEPS:
            # There is no later pre-boundary manager step at which to arm a
            # full clamped window, so arm immediately but keep the same next
            # fixed boundary.
            self._arm_steady_collection(self.prefill_steps + self.sampling_interval)
        else:
            self._reset_route_counters()
            self._set_recording(False)

    def _collect_local_samples(self) -> torch.Tensor:
        counters = [impl.route_counter for impl in self._eplb_impls]
        if any(counter.ndim != 1 or counter.shape[0] != self.num_logical_experts for counter in counters):
            raise RuntimeError("EPLB route counter shape must be [num_logical_experts]")
        return torch.stack(counters).unsqueeze(0).cpu()

    def _publish_expert_load_metrics(self, result):
        if self.global_rank != 0 or "expert_imbalance_ratio" not in result:
            return
        if self.metric_client is None:
            self.metric_client = MetricClient(get_shm_port_args().metric_port)
        self.metric_client.gauge_set(EPLB_EXPERT_IMBALANCE_RATIO_METRIC, result["expert_imbalance_ratio"])

    def _commit_layer_metadata(self, layer_index: int):
        impl = self._eplb_impls[layer_index]
        impl.logical_to_physical_map.copy_(self.target_metadata[layer_index], non_blocking=True)

    def _finish_rebalance(self):
        self.current_placement = self.target_placement
        self.target_placement = None
        self.target_metadata = None
        self.in_flight = False
        self._prepare_next_sampling_window()
        if self.global_rank == 0:
            logger.info(f"eplb completed wall_time={time.time() - self.in_flight_started_at:.2f}s")

    def _poll_in_flight(self):
        local_error = None
        try:
            pending = self.transfer.pending_layers()
        except BaseException as exc:
            pending = []
            local_error = exc
        ready_count = self._control_count(EPLB_CONTROL_ERROR if local_error is not None else len(pending))
        dist.all_reduce(ready_count, op=dist.ReduceOp.MIN, group=self.control_group)
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
            self.transfer.commit(
                layer_index,
                buffer_index,
                lambda: self._commit_layer_metadata(layer_index),
            )
            self.in_flight_layers.pop(0)
        if not self.in_flight_layers:
            self.transfer.finish()
            self._finish_rebalance()

    def _plan_and_broadcast(self, global_load: torch.Tensor):
        """Plan on rank zero and share the serializable result on the evaluation group."""
        result = None
        local_error = None
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
                    candidate = plan_redundant_experts(
                        global_load,
                        self.world_size,
                        self.num_redundant_experts_per_rank,
                        expert_alignment=EPLB_EXPERT_ALIGNMENT,
                        current_placement=self.current_placement,
                        stickiness=self.placement_stickiness,
                    )
                    placement, improved, metrics, before_load, after_load = select_improving_placements(
                        global_load,
                        self.current_placement,
                        candidate,
                        expert_alignment=EPLB_EXPERT_ALIGNMENT,
                        rebalance_gain_threshold=self.rebalance_gain_threshold,
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
                                self.current_placement[layer_index],
                                placement[layer_index],
                            )
                    result = {
                        "kind": ("planned" if bool(torch.any(improved)) else "no_improvement"),
                        "placement": placement,
                        "improved": improved,
                        "before": _imbalance_summary(before_load),
                        "after": _imbalance_summary(after_load),
                        **metrics,
                    }
            except BaseException as exc:
                local_error = exc
                result = {"kind": "error", "message": f"{type(exc).__name__}: {exc}"}
        if self.world_size > 1:
            result_list = [result]
            dist.broadcast_object_list(result_list, src=0, group=self.evaluation_group)
            result = result_list[0]
        if result["kind"] == "error":
            if local_error is not None:
                raise RuntimeError("EPLB planner failed on rank zero") from local_error
            raise RuntimeError(f"EPLB planner failed on rank zero: {result['message']}")
        return result

    def _evaluate_after_event(self, event: torch.cuda.Event):
        """Run the CPU/Gloo planning phase after the frozen CUDA counters are ready."""
        try:
            torch.cuda.set_device(self._eplb_impls[0].route_counter.device)
            event.synchronize()
            local_load = self._collect_local_samples()
            sample_window_steps = (
                self.step_interval
                if self._continuous_collection_end_step is not None
                else self._steady_sample_window_steps()
            )
            # 保留每个当前 rank 的负载，planner 才能准确模拟“本卡优先，
            # 否则在所有远端副本间分配”的运行时路由规则。
            global_load = torch.zeros(
                (*local_load.shape[:2], self.world_size, local_load.shape[2]),
                dtype=local_load.dtype,
            )
            global_load[:, :, self.global_rank] = local_load
            dist.all_reduce(global_load, op=dist.ReduceOp.SUM, group=self.evaluation_group)
            result = self._plan_and_broadcast(global_load)
            result["expert_imbalance_ratio"] = _expert_load_imbalance_ratio(global_load)
            result["sample_window_steps"] = sample_window_steps
            if result["kind"] == "planned":
                metadata = [None] * len(self.weights)
                layer_plans = []
                improved_layer_indices = torch.nonzero(result["improved"], as_tuple=False).flatten()
                if improved_layer_indices.numel():
                    redundant_placements = result["placement"][improved_layer_indices].tolist()
                    num_primary_experts_per_rank = self.num_logical_experts // self.world_size
                    rank_to_logic_expert_ids_by_layer = [
                        [
                            list(
                                range(
                                    rank * num_primary_experts_per_rank,
                                    (rank + 1) * num_primary_experts_per_rank,
                                )
                            )
                            + rank_redundant_expert_ids
                            for rank, rank_redundant_expert_ids in enumerate(layer_placement)
                        ]
                        for layer_placement in redundant_placements
                    ]
                    maps_for_improved_layers = torch.tensor(
                        build_logical_to_physical_maps_for_layers(
                            rank_to_logic_expert_ids_by_layer,
                            self.num_logical_experts,
                            current_rank=self.global_rank,
                        ),
                        dtype=torch.int32,
                    )
                    for improved_layer_offset, layer_index in enumerate(improved_layer_indices.tolist()):
                        placement = result["placement"][layer_index]
                        metadata[layer_index] = maps_for_improved_layers[improved_layer_offset]
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
        self._publish_expert_load_metrics(result)
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
                        "next_sampling_interval=%s sample_window_steps=%s",
                        self.prefill_steps,
                        result["minimum_layer_samples"],
                        result["minimum"],
                        self.sampling_interval,
                        result.get("sample_window_steps"),
                    )
                else:
                    logger.info(
                        "eplb insufficient samples: prefill_steps=%s minimum_layer_samples=%s required=%s "
                        "scheduled_fresh_window_start=%s scheduled_fresh_window_end=%s "
                        "sample_window_steps=%s",
                        self.prefill_steps,
                        result["minimum_layer_samples"],
                        result["minimum"],
                        self._continuous_collection_start_step,
                        self._continuous_collection_end_step,
                        result.get("sample_window_steps"),
                    )
            return False
        if result["kind"] == "no_improvement":
            self.sampling_interval = min(self.sampling_interval * 4, self.step_interval * 16)
            if self.global_rank == 0:
                logger.info(
                    "eplb skip rearrangement: no model improvement model_imbalance_ratio=%.4f "
                    "candidate_model_imbalance_ratio=%.4f candidate_rebalance_gain=%.4f "
                    "candidate_changed_layer_count=%s actual_changed_layer_count=0 next_sampling_interval=%s "
                    "sample_window_steps=%s",
                    result["model_imbalance_ratio"],
                    result["candidate_model_imbalance_ratio"],
                    result["candidate_rebalance_gain"],
                    result["candidate_changed_layer_count"],
                    self.sampling_interval,
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
        self.sampling_interval = self.step_interval
        self._clear_continuous_collection()
        self._reset_route_counters()
        self.target_placement = placement
        self.target_metadata = result["metadata"]
        self.in_flight_layers = [layer_index for layer_index, _ in layer_plans]
        self.in_flight = True
        self.in_flight_started_at = time.time()
        self.transfer.start(layer_plans)
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
                "sample_window_steps=%s",
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
                result.get("sample_window_steps"),
            )


def _imbalance_summary(rank_load: torch.Tensor) -> Dict[str, float]:
    if rank_load.ndim != 3:
        raise ValueError("rank_load must be [samples, layers, ranks]")
    critical = rank_load.max(dim=2).values.sum(dim=0)
    mean = rank_load.mean(dim=2).sum(dim=0)
    layer_imbalance = critical / mean.clamp_min(1.0)
    sorted_imbalance = torch.sort(layer_imbalance).values
    p95_index = max(0, (95 * layer_imbalance.numel() + 99) // 100 - 1)
    return {
        "max": float(layer_imbalance.max().item()),
        "p95": float(sorted_imbalance[p95_index].item()),
    }


def _expert_load_imbalance_ratio(global_load: torch.Tensor) -> float:
    """Average each layer's maximum-to-mean logical-expert token ratio."""
    if global_load.ndim != 4:
        raise ValueError("global_load must be [samples, layers, ranks, logical_experts]")
    if global_load.shape[1] == 0 or global_load.shape[3] == 0:
        raise ValueError("global_load must contain at least one layer and logical expert")
    layer_expert_load = global_load.sum(dim=(0, 2)).to(torch.float64)
    layer_means = layer_expert_load.mean(dim=1)
    valid_layers = layer_means > 0
    if not torch.any(valid_layers):
        return 0.0
    layer_ratios = layer_expert_load.max(dim=1).values[valid_layers] / layer_means[valid_layers]
    return float(layer_ratios.mean().item())


def _find_fused_moe_weights(model):
    weights_by_id = {}
    for layer in model.trans_layers_weight:
        for value in getattr(layer, "__dict__", {}).values():
            if isinstance(value, FusedMoeWeight) and value.enable_ep_moe:
                weights_by_id[id(value)] = value
    return sorted(weights_by_id.values(), key=lambda weight: weight.layer_num_)
