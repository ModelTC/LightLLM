import threading
from typing import Optional

import torch
import torch.distributed as dist

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import find_fused_moe_weights
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.ep_balance import PrefillEPBalanceCounters
from lightllm.distributed import dist_group_manager
from lightllm.utils.device_utils import is_sm100_gpu
from lightllm.utils.dist_utils import (
    get_global_rank,
    get_global_world_size,
)
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)

EP_BALANCE_PREFILL_ROUNDS_PER_REPORT = 100
EP_BALANCE_ROUND_BUFFER_CAPACITY = 4096
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


class EPBalanceMonitor:
    """Report cross-rank imbalance for non-overlapping blocks of complete prefill rounds."""

    def __init__(self, model: TpPartBaseModel):
        self.global_rank = get_global_rank()
        self.world_size = get_global_world_size()
        self.weights = find_fused_moe_weights(model)
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
        # 环形缓冲区
        self._round_buffer = torch.zeros(
            (EP_BALANCE_ROUND_BUFFER_CAPACITY, len(self.weights), 2),
            dtype=torch.int64,
        )
        self._round_lock = threading.Lock()
        self._round_ready = threading.Event()
        self._written_round_count = 0  # Prefill rounds fully written to the ring buffer.
        self._processed_round_count = 0  # Prefill rounds consumed by the monitor thread.

        # Start the first runtime prefill sample from the current cumulative CPU totals.
        self._previous_prefill_totals = self._current_prefill_totals()

        # Reuse the Gloo group created during distributed initialization so all
        # ranks create process groups in the same order before model loading.
        self.gloo_group = dist_group_manager.get_default_group().autotune_group
        threading.Thread(target=self._monitor_loop, daemon=True, name="ep-balance-monitor").start()

    def _current_prefill_totals(self) -> torch.Tensor:
        """Return this rank's cumulative prefill loads as [num_layers, 2]."""
        return torch.tensor(
            [[counter.route_load, counter.compute_load] for counter in self.counters],
            dtype=torch.int64,
        )

    def record_prefill_round(self):
        """Queue one complete all-layer prefill sample from cumulative CPU counters."""
        if not self.enabled:
            return

        with self._round_lock:
            current_totals = self._current_prefill_totals()
            round_load = current_totals - self._previous_prefill_totals
            buffer_index = self._written_round_count % EP_BALANCE_ROUND_BUFFER_CAPACITY
            self._round_buffer[buffer_index].copy_(round_load)
            self._previous_prefill_totals.copy_(current_totals)
            self._written_round_count += 1
            if self._written_round_count - self._processed_round_count >= EP_BALANCE_PREFILL_ROUNDS_PER_REPORT:
                self._round_ready.set()

    def _get_common_round_end(self) -> int:
        """Return the exclusive round boundary completed by every rank."""
        with self._round_lock:
            local_round_end = self._written_round_count

        common_round_end = torch.tensor(local_round_end, dtype=torch.int64)
        dist.all_reduce(common_round_end, op=dist.ReduceOp.MIN, group=self.gloo_group)
        return int(common_round_end.item())

    def _copy_local_rounds(self, start: int, end: int) -> torch.Tensor:
        """Copy local prefill-round loads in the half-open range [start, end)."""
        with self._round_lock:
            start_index = start % EP_BALANCE_ROUND_BUFFER_CAPACITY
            end_index = end % EP_BALANCE_ROUND_BUFFER_CAPACITY
            if start_index < end_index:
                return self._round_buffer[start_index : start_index + (end - start)].clone()
            return torch.cat(
                (
                    self._round_buffer[start_index:],
                    self._round_buffer[:end_index],
                ),
                dim=0,
            )

    def _gather_round_stats(self, local_round_stats: torch.Tensor) -> torch.Tensor:
        """Gather rank-local stats as [round, layer, rank, route/compute]."""
        gathered = [torch.empty_like(local_round_stats) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_round_stats, group=self.gloo_group)
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

        logger.info(
            "ep_balance "
            f"phase=prefill prefill_rounds={compute['prefill_rounds']} "
            "prefill_ep_critical_overhead_gflops_per_routed_token="
            f"{compute['critical_overhead_gflops_per_routed_token']:.4f} "
            "prefill_ep_compute_critical_overhead_ratio="
            f"{compute['prefill_ep_compute_critical_overhead_ratio']:.4f}"
        )

    def _monitor_loop(self):
        """Consume commonly completed rounds in background report-sized windows."""
        try:
            while True:
                self._round_ready.wait()
                common_round_end = self._get_common_round_end()
                if common_round_end - self._processed_round_count > EP_BALANCE_ROUND_BUFFER_CAPACITY:
                    raise RuntimeError("EP balance prefill-round buffer overflowed")

                while common_round_end - self._processed_round_count >= EP_BALANCE_PREFILL_ROUNDS_PER_REPORT:
                    round_start = self._processed_round_count
                    round_end = round_start + EP_BALANCE_PREFILL_ROUNDS_PER_REPORT
                    local_round_stats = self._copy_local_rounds(round_start, round_end)
                    round_stats = self._gather_round_stats(local_round_stats)
                    with self._round_lock:
                        self._processed_round_count = round_end
                    if self.global_rank == 0:
                        self._log_stats(round_stats)

                with self._round_lock:
                    if self._written_round_count - self._processed_round_count < EP_BALANCE_PREFILL_ROUNDS_PER_REPORT:
                        self._round_ready.clear()
        except Exception as exc:
            logger.exception(f"EP balance monitor stopped unexpectedly: {exc}")
            self._disable()
        return

    def _disable(self):
        """Detach counters from MoE weights and disable monitoring."""
        for weight in self.weights:
            weight.fuse_moe_impl.ep_balance_counters = None
        self.enabled = False
