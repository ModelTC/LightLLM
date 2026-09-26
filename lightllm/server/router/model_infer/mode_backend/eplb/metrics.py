"""计算并上报 EPLB 的 logical expert 与 EP rank 负载指标。"""

import numpy as np
import torch

from lightllm.server.metrics.manager import MetricClient

from .placement import ExpertPlacement


COMPUTE_CRITICAL_OVERHEAD_RATIO_BEFORE_REBALANCE_METRIC = (
    "lightllm_prefill_ep_compute_critical_overhead_ratio_before_rebalance"
)
COMPUTE_CRITICAL_OVERHEAD_RATIO_AFTER_REBALANCE_METRIC = (
    "lightllm_prefill_ep_compute_critical_overhead_ratio_after_rebalance"
)
EXPERT_IMBALANCE_RATIO_METRICS = {
    25: "lightllm_eplb_topk_expert_imbalance_ratio_p25",
    50: "lightllm_eplb_topk_expert_imbalance_ratio_p50",
    100: "lightllm_eplb_topk_expert_imbalance_ratio_p100",
}


def logical_expert_imbalance_percentiles(*, expert_load: torch.Tensor) -> dict[int, float]:
    """统计各层 ``最热 logical expert / 本层平均负载`` 的分位数。"""
    assert expert_load.ndim == 2 and expert_load.numel() > 0

    expert_load = expert_load.to(torch.float64)
    mean_load_by_layer = expert_load.mean(dim=1)

    # 没有 token 的层不参与分位数计算，避免产生 0 / 0。
    has_route_load = mean_load_by_layer > 0
    if not torch.any(has_route_load):
        return {percentile: 0.0 for percentile in EXPERT_IMBALANCE_RATIO_METRICS}

    hottest_expert_load = expert_load.max(dim=1).values
    imbalance_ratio_by_layer = hottest_expert_load[has_route_load] / mean_load_by_layer[has_route_load]
    percentiles = tuple(EXPERT_IMBALANCE_RATIO_METRICS)

    # inverted_cdf 就是 nearest-rank 定义；默认 linear 或 nearest 插值都会改变
    # 层数较少时的 P25/P50 语义。
    percentile_values = np.percentile(
        a=imbalance_ratio_by_layer.numpy(),
        q=percentiles,
        method="inverted_cdf",
    )
    return dict(zip(percentiles, percentile_values.tolist()))


def publish_expert_load_metrics(*, metric_client: MetricClient, expert_load: torch.Tensor) -> None:
    """上报 logical expert 层间不均衡分位数。"""
    imbalance_percentiles = logical_expert_imbalance_percentiles(expert_load=expert_load)
    for percentile, metric_name in EXPERT_IMBALANCE_RATIO_METRICS.items():
        metric_client.gauge_set(
            name=metric_name,
            value=imbalance_percentiles[percentile],
        )


def compute_critical_overhead_ratio(
    *,
    logical_expert_load: torch.Tensor,
    placement: ExpertPlacement,
    expert_alignment: int,
) -> float:
    """估算 placement 相对理想 rank 均衡状态的关键路径额外计算比例。

    假设同一 logical expert 的流量由 hash 均匀分配给所有 physical 副本，
    并按 DeepEP 的 expert alignment 对每个副本负载向上取整。每层单独选择
    最繁忙 rank，避免不同层的热点 rank 在汇总时相互抵消。
    """
    assert logical_expert_load.ndim == 2 and logical_expert_load.numel() > 0
    assert expert_alignment > 0

    num_layers, num_logical_experts = logical_expert_load.shape

    # placement: [layer, rank, local physical expert]
    placement_tensor = torch.tensor(placement, dtype=torch.int64)
    assert placement_tensor.ndim == 3 and placement_tensor.shape[0] == num_layers

    # 将每个 physical slot 中保存的 logical expert ID 转成 one-hot：
    # [layer, rank, physical slot] -> [layer, rank, physical slot, logical expert]。
    # 例如 slot 中保存 expert 2，就会在 logical expert 维得到 [0, 0, 1, ...]。
    expert_mask_by_physical_slot = torch.nn.functional.one_hot(
        placement_tensor,
        num_classes=num_logical_experts,
    )

    # 沿 physical slot 维求和，得到每个 rank 持有的专家副本数：[layer, rank, logical expert]。
    replicas_per_rank = expert_mask_by_physical_slot.sum(dim=2).to(torch.float64)

    # 再沿 rank 维求和，得到每个 logical expert 的全局副本数：[layer, logical expert]。
    replica_count_per_expert = replicas_per_rank.sum(dim=1)
    assert torch.all(replica_count_per_expert > 0)

    # logical_expert_load: [layer, logical expert]
    # hash 均匀分流后，每个 physical 副本承担 logical expert 总负载的 1/N。
    load_per_replica = logical_expert_load.to(torch.float64) / replica_count_per_expert
    aligned_load_per_replica = torch.ceil(load_per_replica / expert_alignment) * expert_alignment

    # 广播为 [layer, rank, logical expert] 后沿 expert 维求和，得到每层各 rank
    # 的估算计算量：[layer, rank]。
    estimated_rank_load = (replicas_per_rank * aligned_load_per_replica.unsqueeze(dim=1)).sum(dim=2)

    mean_rank_load_by_layer = estimated_rank_load.mean(dim=1)
    critical_rank_load_by_layer = estimated_rank_load.max(dim=1).values
    total_balanced_compute = mean_rank_load_by_layer.sum()
    if total_balanced_compute == 0:
        return 0.0

    total_critical_overhead = (critical_rank_load_by_layer - mean_rank_load_by_layer).sum()
    return float((total_critical_overhead / total_balanced_compute).item())


def publish_rebalance_compute_metrics(
    *,
    metric_client: MetricClient,
    global_load: torch.Tensor,
    current_placement: ExpertPlacement,
    target_placement: ExpertPlacement,
    expert_alignment: int,
) -> None:
    """使用同一份 planner 输入上报重排前后的关键路径开销。"""
    before_rebalance_ratio = compute_critical_overhead_ratio(
        logical_expert_load=global_load,
        placement=current_placement,
        expert_alignment=expert_alignment,
    )
    after_rebalance_ratio = compute_critical_overhead_ratio(
        logical_expert_load=global_load,
        placement=target_placement,
        expert_alignment=expert_alignment,
    )

    metric_client.gauge_set(
        name=COMPUTE_CRITICAL_OVERHEAD_RATIO_BEFORE_REBALANCE_METRIC,
        value=before_rebalance_ratio,
    )
    metric_client.gauge_set(
        name=COMPUTE_CRITICAL_OVERHEAD_RATIO_AFTER_REBALANCE_METRIC,
        value=after_rebalance_ratio,
    )
