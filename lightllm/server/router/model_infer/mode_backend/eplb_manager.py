import threading
import time
from typing import Dict

import torch
import torch.distributed as dist

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import find_fused_moe_weights
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_placement import (
    build_initial_redundant_expert_ids,
    build_logical_to_physical_map,
    estimate_rank_load,
    plan_redundant_experts,
    select_improving_placements,
)
from lightllm.server.router.model_infer.mode_backend.eplb_transfer import (
    NixlEPLBTransfer,
    build_transfer_plan,
)
from lightllm.utils.dist_utils import get_global_rank, get_global_world_size, get_node_world_size
from lightllm.utils.envs_utils import (
    enable_eplb_rebalance_once,
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


class EPLBManager:
    """Online EPLB with asynchronous GPU expert migration."""

    def __init__(self, model: TpPartBaseModel):
        self.weights = find_fused_moe_weights(model)
        assert self.weights, "EPLB requires at least one EP MoE layer"
        self.global_rank = get_global_rank()
        self.world_size = get_global_world_size()
        self.node_world_size = get_node_world_size()
        self.step_interval = get_prefill_eplb_step_interval()
        self.prefill_cudagraph = model.args.enable_prefill_cudagraph
        self.rebalance_once = enable_eplb_rebalance_once()
        self.prefill_steps = 0
        self.rebalanced = False
        self.initial_sampling_complete = False
        routed = {weight.n_routed_experts for weight in self.weights}
        redundant = {weight.redundancy_expert_num for weight in self.weights}
        assert len(routed) == len(redundant) == 1
        self.num_logical_experts = routed.pop()
        self.redundant_experts_per_rank = redundant.pop()
        initial = build_initial_redundant_expert_ids(
            self.num_logical_experts, self.world_size, self.redundant_experts_per_rank
        )
        self.current_placement = initial.unsqueeze(0).expand(len(self.weights), -1, -1).clone()
        self.in_flight = False
        self.target_placement = None
        self.in_flight_started_at = None
        self.evaluation_in_flight = False
        self._evaluation_lock = threading.Lock()
        self._evaluation_result = None
        self._evaluation_error = None
        self._evaluation_thread = None
        # Before the first sufficient evaluation, collect a dense interval so
        # the planner has representative load.  A planned or no-improvement
        # result switches continuous mode to sparse; insufficient retries dense.
        self._sampling_pending = False
        self._reset_recorded_samples()
        self._set_recording(True)
        self.gloo_group = dist.new_group(list(range(self.world_size)), backend="gloo")
        self.transfer_group = dist.new_group(list(range(self.world_size)), backend="gloo")
        self.transfer = NixlEPLBTransfer(self.weights, self.transfer_group, self.global_rank, self.world_size)
        if self.global_rank == 0:
            logger.info(
                "eplb enabled "
                f"layers={len(self.weights)} logical_experts={self.num_logical_experts} "
                f"redundant_experts_per_rank={self.redundant_experts_per_rank} "
                f"step_interval={self.step_interval} rebalance_once={self.rebalance_once} "
                f"sample_mode={'accumulated' if self.prefill_cudagraph else 'per_dispatch'}"
            )

    def _set_recording(self, enabled: bool):
        for weight in self.weights:
            weight.eplb_experts_record_load_tensor.fill_(int(enabled))
            weight.fuse_moe_impl.eplb_recording = enabled

    def _reset_recorded_samples(self):
        for weight in self.weights:
            weight.routed_expert_counter_tensor.zero_()
            weight.fuse_moe_impl.eplb_recorded_sample_count = 0

    def _sampling_dense(self) -> bool:
        return not self.initial_sampling_complete

    def _prepare_next_sampling_window(self):
        """Reset and arm the next dense or interval-one sampling window."""
        if self.rebalance_once and self.rebalanced:
            self._sampling_pending = False
            self._set_recording(False)
            return
        self._reset_recorded_samples()
        self._sampling_pending = False
        self._set_recording(self._sampling_dense() or self.step_interval == 1)

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
        counters = [weight.routed_expert_counter_tensor for weight in self.weights]
        capacities = [counter.shape[0] for counter in counters]
        if len(set(capacities)) != 1 or any(counter.ndim != 2 for counter in counters):
            raise RuntimeError("EPLB sample capacities differ between layers")
        counter_samples = torch.stack(counters, dim=1)
        if self.prefill_cudagraph:
            metadata = torch.tensor([capacities[0], -capacities[0]], dtype=torch.int64)
            dist.all_reduce(metadata, op=dist.ReduceOp.MIN, group=self.gloo_group)
            if metadata[0] != -metadata[1]:
                raise RuntimeError("EPLB sample capacity differs between ranks")
            return counter_samples.sum(dim=0, keepdim=True).cpu()
        counts = [weight.fuse_moe_impl.eplb_recorded_sample_count for weight in self.weights]
        if len(set(counts)) != 1:
            raise RuntimeError("EPLB recorded sample counts differ between layers")
        sample_count = counts[0]
        metadata = torch.tensor([sample_count, -sample_count, capacities[0], -capacities[0]], dtype=torch.int64)
        dist.all_reduce(metadata, op=dist.ReduceOp.MIN, group=self.gloo_group)
        if metadata[0] != -metadata[1] or metadata[2] != -metadata[3]:
            raise RuntimeError("EPLB recorded sample count or capacity differs between ranks")
        return self._recent_ring_samples(counter_samples, sample_count).cpu()

    def _commit_layer_metadata(self, layer_index: int):
        weight = self.weights[layer_index]
        local_ids = self.target_placement[layer_index, self.global_rank]
        experts_per_rank = self.num_logical_experts // self.world_size
        logical_to_physical, replica_count = build_logical_to_physical_map(
            self.target_placement[layer_index], self.num_logical_experts
        )
        weight.redundancy_expert_ids = local_ids.tolist()
        weight.redundancy_expert_idx_to_local_idx = {
            int(expert): experts_per_rank + slot for slot, expert in enumerate(local_ids.tolist())
        }
        weight.eplb_experts_logical_to_physical_map.copy_(logical_to_physical, non_blocking=True)
        weight.eplb_experts_logical_replica_count.copy_(replica_count, non_blocking=True)

    def _finish_rebalance(self):
        self.current_placement = self.target_placement
        self.target_placement = None
        self.in_flight = False
        self.rebalanced = True
        self._prepare_next_sampling_window()
        if self.global_rank == 0:
            logger.info(f"eplb completed wall_time={time.time() - self.in_flight_started_at:.2f}s")

    def _poll_in_flight(self):
        pending = self.transfer.pending_layers()  # Propagates worker errors on the main thread.
        ready_count = torch.tensor([len(pending)], dtype=torch.int32)
        dist.all_reduce(ready_count, op=dist.ReduceOp.MIN, group=self.gloo_group)
        ready_count = int(ready_count.item())
        if ready_count == 0:
            return
        if ready_count > len(pending) or ready_count > len(self.in_flight_layers):
            raise RuntimeError("EPLB global ready count exceeds the local ordered prefix")
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

    def _evaluate_after_event(self, event: torch.cuda.Event):
        """Run the CPU/Gloo planning phase after the frozen CUDA counters are ready."""
        try:
            torch.cuda.set_device(self.weights[0].routed_expert_counter_tensor.device)
            event.synchronize()
            global_load = self._collect_local_samples()
            dist.all_reduce(global_load, op=dist.ReduceOp.SUM, group=self.gloo_group)
            minimum = self.num_logical_experts * EPLB_MIN_AVG_TOKENS_PER_EXPERT
            layer_samples = global_load.sum(dim=(0, 2))
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
                )
                placement, improved, metrics = select_improving_placements(
                    global_load, self.current_placement, candidate, expert_alignment=EPLB_EXPERT_ALIGNMENT
                )
                before_load = estimate_rank_load(global_load, self.current_placement, EPLB_EXPERT_ALIGNMENT)
                after_load = estimate_rank_load(global_load, placement, EPLB_EXPERT_ALIGNMENT)
                result = {
                    "kind": "planned" if bool(torch.any(improved)) else "no_improvement",
                    "placement": placement,
                    "improved": improved,
                    "before": _imbalance_summary(before_load),
                    "after": _imbalance_summary(after_load),
                    **metrics,
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
            if self.global_rank == 0:
                logger.info(
                    "eplb skip rearrangement: minimum_layer_samples=%s required=%s",
                    result["minimum_layer_samples"],
                    result["minimum"],
                )
            self._prepare_next_sampling_window()
            return False
        if result["kind"] == "no_improvement":
            if self.global_rank == 0:
                logger.info(
                    "eplb skip rearrangement: no model improvement model_imbalance_ratio=%.4f "
                    "candidate_model_imbalance_ratio=%.4f candidate_relative_improvement=%.4f "
                    "candidate_changed_layer_count=%s actual_changed_layer_count=0",
                    result["model_imbalance_ratio"],
                    result["candidate_model_imbalance_ratio"],
                    result["candidate_relative_improvement"],
                    result["candidate_changed_layer_count"],
                )
            self.initial_sampling_complete = True
            self._prepare_next_sampling_window()
            return False
        self._start_rebalance(result)
        return True

    def _start_rebalance(self, result):
        placement, improved = result["placement"], result["improved"]
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
        self.initial_sampling_complete = True
        self._reset_recorded_samples()
        self.target_placement = placement
        self.in_flight_layers = [layer_index for layer_index, _ in layer_plans]
        self.in_flight = True
        self.in_flight_started_at = time.time()
        self.transfer.start(layer_plans)
        if self.global_rank == 0:
            logger.info(
                "eplb started prefill_steps=%s max_before=%.4f max_after=%.4f p95_before=%.4f p95_after=%.4f "
                "model_imbalance_ratio=%.4f candidate_model_imbalance_ratio=%.4f "
                "candidate_relative_improvement=%.4f candidate_changed_layer_count=%s actual_changed_layer_count=%s",
                self.prefill_steps,
                result["before"]["max"],
                result["after"]["max"],
                result["before"]["p95"],
                result["after"]["p95"],
                result["model_imbalance_ratio"],
                result["candidate_model_imbalance_ratio"],
                result["candidate_relative_improvement"],
                result["candidate_changed_layer_count"],
                len(layer_plans),
            )

    def step(self):
        if self.in_flight:
            self._poll_in_flight()
            return
        if self.rebalanced and self.rebalance_once:
            return
        # Always return after polling an existing evaluation.  In particular,
        # interval one must not consume a result and immediately evaluate an
        # empty counter before another forward has been sampled.
        if self.evaluation_in_flight:
            self._poll_evaluation()
            return
        self.prefill_steps += 1
        phase = self.prefill_steps % self.step_interval
        if self._sampling_dense():
            if phase == 0:
                self._start_evaluation()
            return
        if self._sampling_pending:
            self._sampling_pending = False
            self._start_evaluation()
            return
        if self.step_interval == 1:
            self._start_evaluation()
            return
        if phase == self.step_interval - 1:
            self._reset_recorded_samples()
            self._sampling_pending = True
            self._set_recording(True)
