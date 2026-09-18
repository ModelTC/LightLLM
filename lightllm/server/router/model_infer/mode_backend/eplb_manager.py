from concurrent.futures import Future
import threading
import time

import torch
import torch.distributed as dist

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_placement import (
    build_initial_local_expert_ids,
    build_logical_to_physical_maps_for_layers,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.eplb_planner import (
    EPLBPlanner,
    GreedyEPLBPlanner,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import (
    FusedMoeWeight,
)
from lightllm.server.metrics.manager import MetricClient
from lightllm.server.router.model_infer.mode_backend.eplb_transfer import (
    PinnedMemoryEPLBTransfer,
    build_transfer_plan,
)
from lightllm.utils.dist_utils import (
    get_global_rank,
    get_global_world_size,
    get_node_world_size,
)
from lightllm.utils.envs_utils import (
    get_eplb_rebalance_gain_threshold,
    get_prefill_eplb_step_interval,
)
from lightllm.utils.log_utils import init_logger
from lightllm.utils.shm_port_args import get_shm_port_args

logger = init_logger(__name__)
EPLB_EXPERT_ALIGNMENT = 128
EPLB_CONTROL_ERROR = -1
EPLB_EXPERT_IMBALANCE_RATIO_METRIC = "lightllm_eplb_topk_expert_imbalance_ratio"


class EPLBManager:
    """Collect logical load, invoke a planner, and publish migrated rows."""

    def __init__(self, model: TpPartBaseModel):
        weights = _find_fused_moe_weights(model)
        assert weights, "EPLB requires at least one EP MoE layer"
        self.global_rank = get_global_rank()
        self.world_size = get_global_world_size()
        self.node_world_size = get_node_world_size()
        self._eplb_impls = [weight.fuse_moe_impl for weight in weights]

        routed = {impl.n_routed_experts for impl in self._eplb_impls}
        redundant = {impl.num_redundant_experts_per_rank for impl in self._eplb_impls}
        assert len(routed) == len(redundant) == 1
        self.num_logical_experts = routed.pop()
        self.num_redundant_experts_per_rank = redundant.pop()
        self.step_interval = get_prefill_eplb_step_interval()
        self.prefill_steps = 0
        self.next_evaluation_step = self.step_interval

        initial_local_expert_ids = build_initial_local_expert_ids(
            self.num_logical_experts,
            self.world_size,
            self.num_redundant_experts_per_rank,
        )
        num_primary_experts_per_rank = self.num_logical_experts // self.world_size
        initial_redundant_expert_ids = [
            expert_ids[num_primary_experts_per_rank:] for expert_ids in initial_local_expert_ids
        ]
        self.current_placement = torch.tensor(
            [initial_redundant_expert_ids for _ in weights],
            dtype=torch.int64,
        )
        self.planner: EPLBPlanner = GreedyEPLBPlanner(
            self.world_size,
            self.num_redundant_experts_per_rank,
            expert_alignment=EPLB_EXPERT_ALIGNMENT,
            rebalance_gain_threshold=get_eplb_rebalance_gain_threshold(),
        )

        self.in_flight_layers = []
        self.target_placement = None
        self.target_metadata = None
        self._evaluation = None
        self.metric_client = None

        self.evaluation_group = dist.new_group(list(range(self.world_size)), backend="gloo")
        self.control_group = dist.new_group(list(range(self.world_size)), backend="gloo")
        self.transfer_group = dist.new_group(list(range(self.world_size)), backend="gloo")
        self._control_ready_count = torch.empty(1, dtype=torch.int32)
        self.transfer = PinnedMemoryEPLBTransfer(weights, self.transfer_group, self.global_rank)
        self._restart_collection()

        if self.global_rank == 0:
            logger.info(
                f"eplb enabled layers={len(weights)} num_logical_experts={self.num_logical_experts} "
                f"num_redundant_experts_per_rank={self.num_redundant_experts_per_rank} "
                f"step_interval={self.step_interval} planner={type(self.planner).__name__}"
            )

    def poll(self):
        """Advance EPLB only from a globally ordered pre-forward boundary."""
        if self.in_flight_layers:
            self._poll_in_flight()
        elif self._evaluation is not None and self._evaluation_ready_on_all_ranks():
            self._finish_evaluation()

    def step(self):
        if self.in_flight_layers or self._evaluation is not None:
            return
        self.prefill_steps += 1
        if self.prefill_steps >= self.next_evaluation_step:
            self._start_evaluation()

    def _restart_collection(self):
        counters = [impl.route_counter for impl in self._eplb_impls]
        torch._foreach_zero_(counters)
        for impl in self._eplb_impls:
            impl.recording = True
        self.next_evaluation_step = self.prefill_steps + self.step_interval

    def _collect_local_load(self) -> torch.Tensor:
        counters = [impl.route_counter for impl in self._eplb_impls]
        if any(counter.ndim != 1 or counter.shape[0] != self.num_logical_experts for counter in counters):
            raise RuntimeError("EPLB route counter shape must be [num_logical_experts]")
        return torch.stack(counters).cpu()

    def _plan_and_broadcast(self, global_load: torch.Tensor):
        result = None
        local_error = None
        if self.global_rank == 0:
            try:
                result = self.planner.plan(
                    global_load.tolist(),
                    self.current_placement.tolist(),
                ).as_dict()
            except BaseException as exc:
                local_error = exc
                result = {"kind": "error", "message": f"{type(exc).__name__}: {exc}"}
        if self.world_size > 1:
            values = [result]
            dist.broadcast_object_list(values, src=0, group=self.evaluation_group)
            result = values[0]
        if result["kind"] == "error":
            if local_error is not None:
                raise RuntimeError("EPLB planner failed on rank zero") from local_error
            raise RuntimeError(f"EPLB planner failed on rank zero: {result['message']}")
        return result

    def _build_rebalance_data(self, result):
        metadata = {}
        layer_plans = []
        changed_layer_indices = [layer_index for layer_index, changed in enumerate(result["changed_layers"]) if changed]

        num_primary_experts_per_rank = self.num_logical_experts // self.world_size
        rank_to_logic_expert_ids_by_layer = []
        for layer_index in changed_layer_indices:
            rank_to_logic_expert_ids_by_layer.append(
                [
                    list(
                        range(
                            rank * num_primary_experts_per_rank,
                            (rank + 1) * num_primary_experts_per_rank,
                        )
                    )
                    + result["placement"][layer_index][rank]
                    for rank in range(self.world_size)
                ]
            )
        maps = torch.tensor(
            build_logical_to_physical_maps_for_layers(
                rank_to_logic_expert_ids_by_layer,
                self.num_logical_experts,
                current_rank=self.global_rank,
            ),
            dtype=torch.int32,
        )
        for offset, layer_index in enumerate(changed_layer_indices):
            target = torch.tensor(result["placement"][layer_index], dtype=torch.int64)
            metadata[layer_index] = maps[offset]
            layer_plans.append(
                (
                    layer_index,
                    build_transfer_plan(
                        self.current_placement[layer_index],
                        target,
                        self.num_logical_experts,
                        self.world_size,
                        self.node_world_size,
                    ),
                )
            )
        return metadata, layer_plans

    def _evaluate_after_event(self, event: torch.cuda.Event, evaluation: Future):
        try:
            torch.cuda.set_device(self._eplb_impls[0].route_counter.device)
            event.synchronize()
            global_load = self._collect_local_load()
            dist.all_reduce(global_load, op=dist.ReduceOp.SUM, group=self.evaluation_group)
            result = self._plan_and_broadcast(global_load)
            result["expert_imbalance_ratio"] = _expert_load_imbalance_ratio(global_load)
            if result["kind"] == "planned":
                result["metadata"], result["layer_plans"] = self._build_rebalance_data(result)
            evaluation.set_result(result)
        except BaseException as exc:
            evaluation.set_exception(exc)

    def _start_evaluation(self):
        for impl in self._eplb_impls:
            impl.recording = False
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self._evaluation = Future()
        threading.Thread(
            target=self._evaluate_after_event,
            args=(event, self._evaluation),
            daemon=True,
        ).start()

    def _evaluation_ready_on_all_ranks(self) -> bool:
        ready = self._evaluation.done()
        error = self._evaluation.exception() if ready else None
        status = EPLB_CONTROL_ERROR if error is not None else int(ready)
        ready_count = self._control_ready_count.fill_(status)
        dist.all_reduce(ready_count, op=dist.ReduceOp.MIN, group=self.control_group)
        ready_count = int(ready_count.item())
        if ready_count < 0:
            if error is not None:
                raise RuntimeError("EPLB evaluation failed on this rank") from error
            raise RuntimeError("EPLB evaluation failed on another rank")
        return bool(ready_count)

    def _finish_evaluation(self):
        result = self._evaluation.result()
        self._evaluation = None
        self._publish_expert_load_metric(result)
        if result["kind"] != "planned":
            if self.global_rank == 0:
                logger.info("eplb skip rearrangement kind=%s", result["kind"])
            self._restart_collection()
            return
        self._start_rebalance(result)

    def _publish_expert_load_metric(self, result):
        if self.global_rank != 0:
            return
        if self.metric_client is None:
            self.metric_client = MetricClient(get_shm_port_args().metric_port)
        self.metric_client.gauge_set(EPLB_EXPERT_IMBALANCE_RATIO_METRIC, result["expert_imbalance_ratio"])

    def _start_rebalance(self, result):
        self.target_placement = torch.tensor(result["placement"], dtype=torch.int64)
        self.target_metadata = result["metadata"]
        self.in_flight_layers = [layer_index for layer_index, _ in result["layer_plans"]]
        self.in_flight_started_at = time.time()
        self.transfer.start(result["layer_plans"])
        if self.global_rank == 0:
            changed_slots = sum(len(plan) for _, plan in result["layer_plans"])
            logger.info(
                "eplb started prefill_steps=%s max_before=%.4f max_after=%.4f "
                "p95_before=%.4f p95_after=%.4f rebalance_gain=%.4f "
                "changed_layer_count=%s changed_slot_count=%s",
                self.prefill_steps,
                result["before"]["max"],
                result["after"]["max"],
                result["before"]["p95"],
                result["after"]["p95"],
                result["rebalance_gain"],
                result["changed_layer_count"],
                changed_slots,
            )

    def _poll_in_flight(self):
        local_error = None
        try:
            ready_layer = self.transfer.ready_layer()
        except BaseException as exc:
            ready_layer = None
            local_error = exc
        ready_count = self._control_ready_count.fill_(
            EPLB_CONTROL_ERROR if local_error is not None else int(ready_layer is not None)
        )
        dist.all_reduce(ready_count, op=dist.ReduceOp.MIN, group=self.control_group)
        ready_count = int(ready_count.item())
        if ready_count < 0:
            if local_error is not None:
                raise RuntimeError("EPLB transfer worker failed on this rank") from local_error
            raise RuntimeError("EPLB transfer worker failed on another rank")
        if ready_count == 0:
            return
        if ready_layer != self.in_flight_layers[0]:
            raise RuntimeError(f"EPLB ready layer {ready_layer} does not match expected {self.in_flight_layers[0]}")

        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        torch.cuda.current_stream().wait_stream(g_infer_context.get_overlap_stream())
        self.transfer.commit(ready_layer, lambda: self._commit_layer_metadata(ready_layer))
        self.in_flight_layers.pop(0)
        if not self.in_flight_layers:
            self.transfer.finish()
            self._finish_rebalance()

    def _commit_layer_metadata(self, layer_index: int):
        self._eplb_impls[layer_index].logical_to_physical_map.copy_(
            self.target_metadata[layer_index],
            non_blocking=True,
        )

    def _finish_rebalance(self):
        self.current_placement = self.target_placement
        self.target_placement = None
        self.target_metadata = None
        self._restart_collection()
        if self.global_rank == 0:
            logger.info(
                "eplb completed wall_time=%.2fs",
                time.time() - self.in_flight_started_at,
            )


def _expert_load_imbalance_ratio(global_load: torch.Tensor) -> float:
    """Average each layer's maximum-to-mean logical-expert token ratio."""
    if global_load.ndim != 2:
        raise ValueError("global_load must be [layers, logical_experts]")
    global_load = global_load.to(torch.float64)
    layer_means = global_load.mean(dim=1)
    valid_layers = layer_means > 0
    if not torch.any(valid_layers):
        return 0.0
    ratios = global_load.max(dim=1).values[valid_layers] / layer_means[valid_layers]
    return float(ratios.mean().item())


def _find_fused_moe_weights(model):
    weights_by_id = {}
    for layer in model.trans_layers_weight:
        for value in getattr(layer, "__dict__", {}).values():
            if isinstance(value, FusedMoeWeight) and value.enable_ep_moe:
                weights_by_id[id(value)] = value
    return sorted(weights_by_id.values(), key=lambda weight: weight.layer_num_)
