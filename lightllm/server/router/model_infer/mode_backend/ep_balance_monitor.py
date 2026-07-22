import math
import threading
from typing import Dict, List, Optional, TypedDict

import torch
import torch.distributed as dist

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight
from lightllm.utils.dist_utils import get_current_device_id, get_global_rank, get_global_world_size
from lightllm.utils.envs_utils import get_ep_balance_log_interval
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)

EP_BALANCE_MIN_AVG_TOKENS_PER_EXPERT = 100
PHASE_NAMES = ("prefill", "decode")


class RankLoadSummary(TypedDict):
    imbalance_max: float
    imbalance_p95: float
    rank_load_normalized: List[float]


def _summarize_rank_load(rank_load: torch.Tensor) -> RankLoadSummary:
    """Summarize a [layer, rank] load matrix."""
    rank_load = rank_load.to(torch.float64)
    layer_mean = rank_load.mean(dim=1)
    layer_imbalance = rank_load.max(dim=1).values / layer_mean
    sorted_layer_indices = torch.argsort(layer_imbalance, stable=True)
    p95_rank = math.ceil(0.95 * layer_imbalance.numel()) - 1
    p95_layer_index = int(sorted_layer_indices[p95_rank].item())
    p95_layer_rank_load = rank_load[p95_layer_index]
    normalized_rank_load = p95_layer_rank_load / p95_layer_rank_load.sum()
    return {
        "imbalance_max": float(layer_imbalance.max().item()),
        "imbalance_p95": float(layer_imbalance[p95_layer_index].item()),
        "rank_load_normalized": normalized_rank_load.tolist(),
    }


def calculate_ep_balance_stats(
    window_stats: torch.Tensor,
    layer_ids: torch.Tensor,
    routed_expert_nums: torch.Tensor,
    min_avg_tokens_per_expert: int = EP_BALANCE_MIN_AVG_TOKENS_PER_EXPERT,
) -> Dict[str, Optional[Dict[str, RankLoadSummary]]]:
    """Calculate per-phase summaries from [rank, layer, phase, route/compute]."""
    assert window_stats.ndim == 4 and tuple(window_stats.shape[2:]) == (2, 2)
    assert window_stats.shape[1] == layer_ids.numel() == routed_expert_nums.numel()

    results: Dict[str, Optional[Dict[str, RankLoadSummary]]] = {}
    for phase, phase_name in enumerate(PHASE_NAMES):
        route_rank_load = window_stats[:, :, phase, 0].transpose(0, 1)
        compute_rank_load = window_stats[:, :, phase, 1].transpose(0, 1)
        layer_route_samples = route_rank_load.sum(dim=1)
        minimum_samples = routed_expert_nums * min_avg_tokens_per_expert

        # All MoE layers should see the same routed-token count. Treat the phase as
        # one window so a partially captured forward cannot enter the summaries.
        if torch.any(layer_route_samples < minimum_samples):
            results[phase_name] = None
            continue

        results[phase_name] = {
            "compute": _summarize_rank_load(compute_rank_load),
        }
    return results


class EPBalanceMonitor:
    """Periodically aggregate cumulative per-layer EP counters and log window summaries."""

    def __init__(self, model: TpPartBaseModel):
        self.interval = get_ep_balance_log_interval()
        self.global_rank = get_global_rank()
        self.world_size = get_global_world_size()
        self.device_id = get_current_device_id()
        self.weights = self._find_ep_weights(model)
        self.thread: Optional[threading.Thread] = None

        if self.interval == 0 or not self.weights:
            return

        self.layer_ids = torch.tensor([weight.layer_num_ for weight in self.weights], dtype=torch.int64)
        self.routed_expert_nums = torch.tensor([weight.n_routed_experts for weight in self.weights], dtype=torch.int64)
        # Keep monitoring collectives isolated from runtime Triton autotuning and
        # inference NCCL collectives.
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

    def _snapshot_local_counters(self) -> torch.Tensor:
        # The barrier is infrequent and gives the snapshot a coherent boundary
        # across the model's overlap CUDA streams.
        torch.cuda.synchronize(self.device_id)
        counters = torch.stack([weight.ep_balance_counters for weight in self.weights])
        return counters.cpu()

    def _gather_window_stats(self, local_window_stats: torch.Tensor) -> torch.Tensor:
        gathered = [torch.empty_like(local_window_stats) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_window_stats, group=self.gloo_group)
        return torch.stack(gathered)

    def _log_stats(self, window_stats: torch.Tensor):
        summaries = calculate_ep_balance_stats(window_stats, self.layer_ids, self.routed_expert_nums)
        for phase_name, phase_stats in summaries.items():
            if phase_stats is None:
                continue
            compute = phase_stats["compute"]
            normalized_rank_load = ",".join(f"{value:.4f}" for value in compute["rank_load_normalized"])
            logger.info(
                "ep_balance "
                f"phase={phase_name} interval={self.interval} "
                f"compute_imbalance_max={compute['imbalance_max']:.4f} "
                f"compute_imbalance_p95={compute['imbalance_p95']:.4f} "
                "compute_rank_load_normalized="
                f"[{normalized_rank_load}]"
            )

    def _monitor_loop(self):
        try:
            torch.cuda.set_device(self.device_id)
            previous = self._snapshot_local_counters()
            wait_event = threading.Event()
            while True:
                wait_event.wait(self.interval)
                current = self._snapshot_local_counters()
                local_window_stats = current - previous
                previous = current
                window_stats = self._gather_window_stats(local_window_stats)
                if self.global_rank == 0:
                    self._log_stats(window_stats)
        except BaseException as exc:
            logger.exception(f"EP balance monitor stopped unexpectedly: {exc}")
