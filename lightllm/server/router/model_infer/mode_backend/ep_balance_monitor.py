import threading
from typing import List, Optional, TypedDict

import torch
import torch.distributed as dist

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import (
    FusedMoeWeight,
)
from lightllm.utils.dist_utils import (
    get_current_device_id,
    get_global_rank,
    get_global_world_size,
)
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)

EP_BALANCE_MIN_AVG_TOKENS_PER_EXPERT = 100
EP_BALANCE_PREFILL_ROUNDS_PER_REPORT = 100
EP_BALANCE_ROUND_BUFFER_CAPACITY = 4096
PREFILL_PHASE = 0
ROUTE_LOAD = 0
COMPUTE_LOAD = 1
FLOPS_PER_MATMUL_MAC = 2
MOE_MATMULS_PER_EXPERT_TOKEN = 3
FLOPS_PER_GFLOP = 1_000_000_000


class PrefillBalanceSummary(TypedDict):
    prefill_rounds: int
    critical_overhead_gflops_per_token: float
    prefill_ep_compute_critical_overhead_ratio: float


def calculate_prefill_balance_stats(
    round_stats: torch.Tensor,
    total_routed_experts: int,
    layer_compute_flops: torch.Tensor,
    layer_topks: torch.Tensor,
    min_avg_tokens_per_expert: int = EP_BALANCE_MIN_AVG_TOKENS_PER_EXPERT,
) -> Optional[PrefillBalanceSummary]:
    """Summarize complete-prefill samples from [round, layer, rank, route/compute].

    MoE layers execute sequentially, and every layer waits for its slowest EP
    rank.  Preserve the layer dimension until after taking the cross-rank max so
    that different slow ranks in different layers cannot cancel each other.
    """
    assert round_stats.ndim == 4 and round_stats.shape[3] == 2
    assert total_routed_experts > 0
    assert layer_compute_flops.ndim == layer_topks.ndim == 1
    assert round_stats.shape[1] == layer_compute_flops.numel() == layer_topks.numel()
    assert torch.all(layer_compute_flops > 0) and torch.all(layer_topks > 0)

    minimum_route_samples = total_routed_experts * min_avg_tokens_per_expert
    total_route_load = round_stats[:, :, :, ROUTE_LOAD].sum()
    if total_route_load < minimum_route_samples:
        return None

    compute_rank_load = round_stats[:, :, :, COMPUTE_LOAD]
    compute_rank_load_float = compute_rank_load.to(torch.float64)
    excess_compute_load = compute_rank_load_float.max(dim=2).values - compute_rank_load_float.mean(dim=2)

    # Every MoE expert token executes the two projections packed in w13 plus
    # the w2 projection.  Weight each padded compute token by the layer's
    # actual matrix sizes so the metric remains comparable across models.
    excess_compute_flops = (excess_compute_load * layer_compute_flops.to(torch.float64)).sum()
    balanced_compute_flops = (compute_rank_load_float.mean(dim=2) * layer_compute_flops.to(torch.float64)).sum()
    if balanced_compute_flops == 0:
        return None

    # Each source token contributes top-k routed assignments in every layer.
    # Summing route load over all layers and dividing by the sum of layer top-k
    # values recovers the number of source tokens represented by this block.
    source_tokens = total_route_load.to(torch.float64) / layer_topks.to(torch.float64).sum()
    if source_tokens == 0:
        return None

    return {
        "prefill_rounds": int(compute_rank_load.shape[0]),
        "critical_overhead_gflops_per_token": float((excess_compute_flops / source_tokens / FLOPS_PER_GFLOP).item()),
        "prefill_ep_compute_critical_overhead_ratio": float((excess_compute_flops / balanced_compute_flops).item()),
    }


class EPBalanceMonitor:
    """Report cross-rank imbalance for non-overlapping blocks of complete prefill rounds."""

    def __init__(self, model: TpPartBaseModel):
        self.global_rank = get_global_rank()
        self.world_size = get_global_world_size()
        self.device_id = get_current_device_id()
        self.weights = self._find_ep_weights(model)
        self.enabled = bool(self.weights)
        self.thread: Optional[threading.Thread] = None

        if not self.enabled:
            return

        self.total_routed_experts = sum(weight.n_routed_experts for weight in self.weights)
        self.layer_compute_flops = torch.tensor(
            [
                FLOPS_PER_MATMUL_MAC * MOE_MATMULS_PER_EXPERT_TOKEN * weight.hidden_size * weight.moe_intermediate_size
                for weight in self.weights
            ],
            dtype=torch.float64,
        )
        self.layer_topks = torch.tensor(
            [weight.num_experts_per_tok for weight in self.weights],
            dtype=torch.float64,
        )
        self._round_buffer = torch.zeros(
            (EP_BALANCE_ROUND_BUFFER_CAPACITY, len(self.weights), 2),
            dtype=torch.int64,
            device="cuda",
        )
        self._round_lock = threading.Lock()
        self._round_ready = threading.Event()
        self._round_write_sequence = 0
        self._round_read_sequence = 0

        # Model initialization and CUDA-graph capture may already have touched the
        # cumulative counters. Start the first runtime prefill sample from here.
        torch.cuda.synchronize(self.device_id)
        self._previous_prefill_totals = self._current_prefill_totals()
        torch.cuda.synchronize(self.device_id)

        # Keep monitoring collectives isolated from inference NCCL collectives.
        self.gloo_group = dist.new_group(list(range(self.world_size)), backend="gloo")
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True, name="ep-balance-monitor")
        self.thread.start()

    @staticmethod
    def _find_ep_weights(model: TpPartBaseModel) -> List[FusedMoeWeight]:
        weights = []
        for layer in model.trans_layers_weight:
            for value in vars(layer).values():
                if isinstance(value, FusedMoeWeight) and value.enable_ep_moe:
                    weights.append(value)
        return sorted(weights, key=lambda weight: weight.layer_num_)

    def _current_prefill_totals(self) -> torch.Tensor:
        counters = torch.stack([weight.ep_balance_counters for weight in self.weights])
        return counters[:, PREFILL_PHASE, :]

    @torch.no_grad()
    def record_prefill_round(self):
        """Queue one all-layer prefill sample on the inference stream without a CPU sync."""
        if not self.enabled:
            return

        with self._round_lock:
            current_totals = self._current_prefill_totals()
            round_load = current_totals - self._previous_prefill_totals
            buffer_index = self._round_write_sequence % EP_BALANCE_ROUND_BUFFER_CAPACITY
            self._round_buffer[buffer_index].copy_(round_load)
            self._previous_prefill_totals.copy_(current_totals)
            self._round_write_sequence += 1
            if self._round_write_sequence - self._round_read_sequence >= EP_BALANCE_PREFILL_ROUNDS_PER_REPORT:
                self._round_ready.set()

    def _get_common_round_end(self) -> int:
        # Synchronization establishes a coherent boundary for samples queued on
        # the model's overlap CUDA stream.
        with self._round_lock:
            torch.cuda.synchronize(self.device_id)
            local_round_end = self._round_write_sequence

        common_round_end = torch.tensor(local_round_end, dtype=torch.int64)
        dist.all_reduce(common_round_end, op=dist.ReduceOp.MIN, group=self.gloo_group)
        return int(common_round_end.item())

    def _copy_local_rounds(self, start: int, end: int) -> torch.Tensor:
        with self._round_lock:
            torch.cuda.synchronize(self.device_id)
            start_index = start % EP_BALANCE_ROUND_BUFFER_CAPACITY
            end_index = end % EP_BALANCE_ROUND_BUFFER_CAPACITY
            if start_index < end_index:
                return self._round_buffer[start_index : start_index + (end - start)].cpu()
            return torch.cat(
                (
                    self._round_buffer[start_index:],
                    self._round_buffer[:end_index],
                ),
                dim=0,
            ).cpu()

    def _gather_round_stats(self, local_round_stats: torch.Tensor) -> torch.Tensor:
        gathered = [torch.empty_like(local_round_stats) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_round_stats, group=self.gloo_group)
        # [rank, round, layer, route/compute]
        #     -> [round, layer, rank, route/compute]
        return torch.stack(gathered).permute(1, 2, 0, 3)

    def _log_stats(self, round_stats: torch.Tensor):
        compute = calculate_prefill_balance_stats(
            round_stats,
            self.total_routed_experts,
            self.layer_compute_flops,
            self.layer_topks,
        )
        if compute is None:
            return

        logger.info(
            "ep_balance "
            f"phase=prefill prefill_rounds={compute['prefill_rounds']} "
            "prefill_ep_critical_overhead_gflops_per_token="
            f"{compute['critical_overhead_gflops_per_token']:.4f} "
            "prefill_ep_compute_critical_overhead_ratio="
            f"{compute['prefill_ep_compute_critical_overhead_ratio']:.4f}"
        )

    def _monitor_loop(self):
        try:
            torch.cuda.set_device(self.device_id)
            while True:
                self._round_ready.wait()
                common_round_end = self._get_common_round_end()
                if common_round_end - self._round_read_sequence > EP_BALANCE_ROUND_BUFFER_CAPACITY:
                    raise RuntimeError("EP balance prefill-round buffer overflowed")

                while common_round_end - self._round_read_sequence >= EP_BALANCE_PREFILL_ROUNDS_PER_REPORT:
                    round_start = self._round_read_sequence
                    round_end = round_start + EP_BALANCE_PREFILL_ROUNDS_PER_REPORT
                    local_round_stats = self._copy_local_rounds(round_start, round_end)
                    round_stats = self._gather_round_stats(local_round_stats)
                    with self._round_lock:
                        self._round_read_sequence = round_end
                    if self.global_rank == 0:
                        self._log_stats(round_stats)

                with self._round_lock:
                    if self._round_write_sequence - self._round_read_sequence < EP_BALANCE_PREFILL_ROUNDS_PER_REPORT:
                        self._round_ready.clear()
        except BaseException as exc:
            logger.exception(f"EP balance monitor stopped unexpectedly: {exc}")
        return
