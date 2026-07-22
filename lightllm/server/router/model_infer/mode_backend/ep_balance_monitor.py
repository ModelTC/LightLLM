import threading
from array import array
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.ep_balance import PrefillEPBalanceCounters
from lightllm.distributed.communication_op import dist_group_manager
from lightllm.server.metrics.manager import MetricClient
from lightllm.utils.device_utils import is_sm100_gpu
from lightllm.utils.dist_utils import (
    get_global_rank,
    get_global_world_size,
)
from lightllm.utils.log_utils import init_logger
from lightllm.utils.shm_port_args import get_shm_port_args


logger = init_logger(__name__)

EP_BALANCE_PREFILL_ROUNDS_PER_REPORT = 100
EP_BALANCE_ROUND_BUFFER_CAPACITY = 4096
EP_BALANCE_PRESSURE_DRIFT_BUCKET_ROUNDS = 20
EP_BALANCE_PRESSURE_DRIFT_STABLE_THRESHOLD = 0.10
ROUTE_LOAD = 0
COMPUTE_LOAD = 1
GFLOP = 1_000_000_000


def should_enable_ep_balance_monitor(args) -> bool:
    if args.enable_prefill_cudagraph or is_sm100_gpu():
        return False
    return args.enable_ep_moe and not args.disable_ep_balance_monitor and args.run_mode != "decode"


def calculate_prefill_balance_stats(
    round_stats: torch.Tensor,  # [num_rounds, num_layers, world_size, 2] (route/compute)
    layer_routed_experts: torch.Tensor,  # [num_layers]
    layer_flops_per_expert_token: torch.Tensor,  # [num_layers]
    layer_topks: torch.Tensor,  # [num_layers]
    source_token_replication: int,
    report_min_route_samples_per_expert: int = 100,
) -> Optional[dict]:
    """Summarize complete-prefill samples from [round, layer, rank, route/compute].

    MoE layers execute sequentially, and every layer waits for its slowest EP
    rank.  Preserve the layer dimension until after taking the cross-rank max so
    that different slow ranks in different layers cannot cancel each other.
    """
    assert source_token_replication > 0

    layer_route_load = round_stats[:, :, :, ROUTE_LOAD].sum(dim=(0, 2))
    minimum_route_samples = layer_routed_experts * report_min_route_samples_per_expert
    if torch.any(layer_route_load < minimum_route_samples):
        return None
    total_route_load = layer_route_load.sum()

    compute_rank_load = round_stats[:, :, :, COMPUTE_LOAD]
    compute_rank_load_float = compute_rank_load.to(torch.float64)
    excess_compute_load = compute_rank_load_float.max(dim=2).values - compute_rank_load_float.mean(dim=2)

    # Every MoE expert token executes the two projections packed in w13 plus
    # the w2 projection.  Weight each padded compute token by the layer's
    # actual matrix sizes so the metric remains comparable across models.
    excess_compute_flops = (excess_compute_load * layer_flops_per_expert_token.to(torch.float64)).sum()
    balanced_compute_flops = (
        compute_rank_load_float.mean(dim=2) * layer_flops_per_expert_token.to(torch.float64)
    ).sum()
    if balanced_compute_flops == 0:
        return None

    # Non-TPSP prefill gathers one route-load copy per TP rank.  Divide the
    # replica count out so GFLOP/token uses logical source tokens.
    source_tokens = total_route_load.to(torch.float64) / (
        layer_topks.to(torch.float64).sum() * source_token_replication
    )
    if source_tokens == 0:
        return None

    return {
        "prefill_rounds": int(compute_rank_load.shape[0]),
        "critical_overhead_gflops_per_routed_token": float((excess_compute_flops / source_tokens / GFLOP).item()),
        "prefill_ep_compute_critical_overhead_ratio": float((excess_compute_flops / balanced_compute_flops).item()),
    }


def calculate_prefill_placement_pressure_drift(
    round_stats: torch.Tensor,
    previous_pressure_signature: Optional[torch.Tensor] = None,
    bucket_rounds: int = EP_BALANCE_PRESSURE_DRIFT_BUCKET_ROUNDS,
) -> Tuple[float, torch.Tensor]:
    """Measure how prefill rank-pressure placement changes across time buckets.

    This is rank-0 CPU-only report analysis.  The returned final bucket is a
    compact signature that allows the next report to include the boundary pair.
    """
    if bucket_rounds <= 0:
        raise ValueError(f"bucket_rounds must be positive, got {bucket_rounds}")
    if round_stats.ndim != 4 or round_stats.shape[-1] != 2:
        raise ValueError(
            "round_stats must have shape [num_rounds, num_layers, world_size, 2], " f"got {tuple(round_stats.shape)}"
        )
    num_rounds, num_layers, world_size, _ = round_stats.shape
    if num_rounds <= 0:
        raise ValueError("round_stats must contain at least one round")
    if num_layers <= 0 or world_size <= 0:
        raise ValueError("round_stats must contain at least one layer and rank")
    if num_rounds % bucket_rounds != 0:
        raise ValueError(f"num_rounds ({num_rounds}) must be divisible by bucket_rounds ({bucket_rounds})")

    num_buckets = num_rounds // bucket_rounds
    bucket_rank_load = (
        round_stats[:, :, :, COMPUTE_LOAD]
        .to(torch.float64)
        .reshape(num_buckets, bucket_rounds, num_layers, world_size)
        .sum(dim=1)
    )
    mean_rank_load = bucket_rank_load.mean(dim=2, keepdim=True).clamp_min(1)
    pressure = torch.relu(bucket_rank_load / mean_rank_load - 1)

    if previous_pressure_signature is not None:
        expected_shape = (num_layers, world_size)
        if tuple(previous_pressure_signature.shape) != expected_shape:
            raise ValueError(
                "previous_pressure_signature must have shape "
                f"{expected_shape}, got {tuple(previous_pressure_signature.shape)}"
            )
        left = torch.cat((previous_pressure_signature.to(torch.float64).unsqueeze(0), pressure[:-1]), dim=0)
        right = pressure
    else:
        left = pressure[:-1]
        right = pressure[1:]

    total_pressure = (left + right).sum()
    if total_pressure == 0:
        drift = 0.0
    else:
        drift = float((left - right).abs().sum().div(total_pressure).item())
    return drift, pressure[-1].clone()


def classify_prefill_placement_pressure_drift(drift: float) -> str:
    if drift < EP_BALANCE_PRESSURE_DRIFT_STABLE_THRESHOLD:
        return "stable"
    return "dynamic"


def _find_fused_moe_weights(model):
    weights_by_id = {}
    for layer in model.trans_layers_weight:
        for value in getattr(layer, "__dict__", {}).values():
            if isinstance(value, FusedMoeWeight) and value.enable_ep_moe:
                weights_by_id[id(value)] = value
    return sorted(weights_by_id.values(), key=lambda weight: weight.layer_num_)


class EPBalanceMonitor:
    """Report cross-rank imbalance for non-overlapping blocks of complete prefill rounds."""

    def __init__(self, model: TpPartBaseModel):
        self.global_rank = get_global_rank()
        self.world_size = get_global_world_size()
        self.weights = _find_fused_moe_weights(model)
        self.enabled = bool(self.weights)

        if not self.enabled:
            return

        self.source_token_replication = 1 if model.args.enable_tpsp_mix_mode else model.tp_world_size_
        self.counters: list[PrefillEPBalanceCounters] = [PrefillEPBalanceCounters() for _ in self.weights]
        for weight, counter in zip(self.weights, self.counters):
            weight.fuse_moe_impl.ep_balance_counters = counter
        self.layer_routed_experts = torch.tensor(
            [weight.n_routed_experts for weight in self.weights], dtype=torch.int64
        )
        self.layer_flops_per_expert_token = torch.tensor(
            [
                # Each expert-token performs gate, up, and down projections; each MAC counts as 2 FLOPs.
                2 * 3 * weight.hidden_size * weight.moe_intermediate_size
                for weight in self.weights
            ],
            dtype=torch.float64,
        )
        self.layer_topks = torch.tensor(
            [weight.num_experts_per_tok for weight in self.weights],
            dtype=torch.float64,
        )
        self._round_buffer_storage = array("q", [0]) * (EP_BALANCE_ROUND_BUFFER_CAPACITY * len(self.weights) * 2)
        self._round_buffer = torch.frombuffer(self._round_buffer_storage, dtype=torch.int64).view(
            EP_BALANCE_ROUND_BUFFER_CAPACITY, len(self.weights), 2
        )
        self._round_ready = threading.Event()
        self._written_round_count = 0  # Prefill rounds fully written to the ring buffer.
        self._processed_round_count = 0  # Prefill rounds consumed by the monitor thread.
        self._overflowed = False
        self._previous_pressure_signature: Optional[torch.Tensor] = None
        self._common_round_end = torch.zeros((), dtype=torch.int64)

        self.gloo_group = dist_group_manager.ep_balance_monitor_group
        if self.gloo_group is None:
            raise RuntimeError("EP balance monitor requires a pre-created dedicated Gloo process group")
        self.metric_client = MetricClient(get_shm_port_args().metric_port) if self.global_rank == 0 else None
        threading.Thread(target=self._monitor_loop, daemon=True, name="ep-balance-monitor").start()

    def record_prefill_round(self):
        """Publish one complete all-layer prefill sample to the SPSC ring."""
        if not self.enabled:
            return

        written_round_count = self._written_round_count
        if written_round_count - self._processed_round_count >= EP_BALANCE_ROUND_BUFFER_CAPACITY:
            if not self._overflowed:
                self._overflowed = True
                self._round_ready.set()
            return

        storage_index = (written_round_count % EP_BALANCE_ROUND_BUFFER_CAPACITY) * len(self.counters) * 2
        for counter in self.counters:
            self._round_buffer_storage[storage_index] = counter.route_load
            self._round_buffer_storage[storage_index + 1] = counter.compute_load
            counter.route_load = 0
            counter.compute_load = 0
            storage_index += 2

        # Publish only after the entire slot is written.  The SPSC producer and
        # monitor thread run under the CPython GIL, so this count is the release
        # point for the corresponding ring slot.
        self._written_round_count = written_round_count + 1
        if self._written_round_count - self._processed_round_count >= EP_BALANCE_PREFILL_ROUNDS_PER_REPORT:
            self._round_ready.set()

    def _get_common_round_end(self) -> int:
        """Return the exclusive round boundary completed by every rank."""
        self._common_round_end.fill_(self._written_round_count)
        dist.all_reduce(self._common_round_end, op=dist.ReduceOp.MIN, group=self.gloo_group)
        return int(self._common_round_end.item())

    def _raise_buffer_overflow(self, phase: str, common_round_end: Optional[int] = None):
        message = (
            "EP balance prefill-round buffer overflowed "
            f"phase={phase} written={self._written_round_count} "
            f"processed={self._processed_round_count} capacity={EP_BALANCE_ROUND_BUFFER_CAPACITY}"
        )
        if common_round_end is not None:
            message += f" common_round_end={common_round_end}"
        raise RuntimeError(message)

    def _copy_local_rounds(self, start: int, end: int) -> torch.Tensor:
        """Copy local prefill-round loads in the half-open range [start, end)."""
        num_rounds = end - start
        if num_rounds > EP_BALANCE_ROUND_BUFFER_CAPACITY:
            raise ValueError("requested EP balance round range exceeds ring capacity")
        start_index = start % EP_BALANCE_ROUND_BUFFER_CAPACITY
        if start_index + num_rounds <= EP_BALANCE_ROUND_BUFFER_CAPACITY:
            return self._round_buffer[start_index : start_index + num_rounds].clone()
        end_index = (start_index + num_rounds) % EP_BALANCE_ROUND_BUFFER_CAPACITY
        return torch.cat((self._round_buffer[start_index:], self._round_buffer[:end_index]), dim=0)

    def _gather_round_stats(self, local_round_stats: torch.Tensor) -> Optional[torch.Tensor]:
        """Gather rank-local stats as [round, layer, rank, route/compute]."""
        gathered = (
            [torch.empty_like(local_round_stats) for _ in range(self.world_size)] if self.global_rank == 0 else None
        )
        dist.gather(local_round_stats, gather_list=gathered, dst=0, group=self.gloo_group)
        if self.global_rank != 0:
            return None
        # [rank, round, layer, route/compute]
        #     -> [round, layer, rank, route/compute]
        return torch.stack(gathered).permute(1, 2, 0, 3)

    def _log_stats(self, round_stats: torch.Tensor):
        """Compute and log balance statistics for one complete global window."""
        compute = calculate_prefill_balance_stats(
            round_stats,
            self.layer_routed_experts,
            self.layer_flops_per_expert_token,
            self.layer_topks,
            self.source_token_replication,
        )
        if compute is None:
            return

        drift, self._previous_pressure_signature = calculate_prefill_placement_pressure_drift(
            round_stats,
            previous_pressure_signature=self._previous_pressure_signature,
        )
        drift_state = classify_prefill_placement_pressure_drift(drift)

        logger.info(
            "ep_balance "
            f"phase=prefill prefill_rounds={compute['prefill_rounds']} "
            "prefill_ep_critical_overhead_gflops_per_routed_token="
            f"{compute['critical_overhead_gflops_per_routed_token']:.4f} "
            "prefill_ep_compute_critical_overhead_ratio="
            f"{compute['prefill_ep_compute_critical_overhead_ratio']:.4f} "
            f"prefill_ep_placement_pressure_drift={drift:.4f} "
            f"prefill_ep_placement_pressure_state={drift_state}"
        )
        self.metric_client.gauge_set(
            "lightllm_prefill_ep_critical_overhead_gflops_per_routed_token",
            compute["critical_overhead_gflops_per_routed_token"],
        )
        self.metric_client.gauge_set(
            "lightllm_prefill_ep_compute_critical_overhead_ratio",
            compute["prefill_ep_compute_critical_overhead_ratio"],
        )
        self.metric_client.gauge_set("lightllm_prefill_ep_placement_pressure_drift", drift)

    def _monitor_loop(self):
        """Consume commonly completed rounds in background report-sized windows."""
        try:
            while True:
                self._round_ready.wait()
                self._round_ready.clear()
                if self._overflowed:
                    self._raise_buffer_overflow("before_sync")
                common_round_end = self._get_common_round_end()
                if common_round_end - self._processed_round_count > EP_BALANCE_ROUND_BUFFER_CAPACITY:
                    self._raise_buffer_overflow("common_round_lag", common_round_end=common_round_end)

                while common_round_end - self._processed_round_count >= EP_BALANCE_PREFILL_ROUNDS_PER_REPORT:
                    round_start = self._processed_round_count
                    round_end = round_start + EP_BALANCE_PREFILL_ROUNDS_PER_REPORT
                    local_round_stats = self._copy_local_rounds(round_start, round_end)
                    if self._overflowed:
                        self._raise_buffer_overflow("after_copy")
                    round_stats = self._gather_round_stats(local_round_stats)
                    self._processed_round_count = round_end
                    if self.global_rank == 0:
                        self._log_stats(round_stats)

                if self._written_round_count - self._processed_round_count >= EP_BALANCE_PREFILL_ROUNDS_PER_REPORT:
                    self._round_ready.set()
        except Exception as exc:
            logger.exception(f"EP balance monitor stopped unexpectedly: {exc}")
            self._disable()
        return

    def _disable(self):
        """Detach counters from MoE weights and disable monitoring."""
        for weight in self.weights:
            weight.fuse_moe_impl.ep_balance_counters = None
        self.enabled = False
