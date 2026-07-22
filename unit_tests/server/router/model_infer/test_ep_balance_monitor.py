import pytest
import torch

from lightllm.common.basemodel.triton_kernel.fused_moe.ep_balance import (
    accumulate_ep_balance,
)
from lightllm.server.router.model_infer.mode_backend.ep_balance_monitor import (
    calculate_prefill_balance_stats,
)


def test_calculate_prefill_balance_stats_over_complete_rounds():
    # [prefill_round, layer, rank, route/compute]
    stats = torch.zeros((2, 2, 4, 2), dtype=torch.int64)
    stats[:, :, :, 0] = 20
    stats[:, :, :, 1] = torch.tensor(
        [
            [
                [100, 100, 100, 100],
                [100, 100, 100, 200],
            ],
            [
                [100, 200, 100, 100],
                [100, 100, 100, 100],
            ],
        ],
        dtype=torch.int64,
    )

    result = calculate_prefill_balance_stats(
        stats,
        total_routed_experts=4,
        layer_compute_flops=torch.tensor([1e9, 3e9]),
        layer_topks=torch.tensor([1, 1]),
        min_avg_tokens_per_expert=10,
    )

    assert result["prefill_rounds"] == 2
    assert result["critical_overhead_gflops_per_token"] == pytest.approx(1.875)
    assert result["prefill_ep_compute_critical_overhead_ratio"] == pytest.approx(1 / 3)


def test_different_slowest_ranks_do_not_cancel_across_layers():
    stats = torch.zeros((1, 2, 2, 2), dtype=torch.int64)
    stats[:, :, :, 0] = 100
    stats[0, :, :, 1] = torch.tensor(
        [
            [200, 100],
            [100, 200],
        ],
        dtype=torch.int64,
    )

    result = calculate_prefill_balance_stats(
        stats,
        total_routed_experts=2,
        layer_compute_flops=torch.tensor([1e9, 2e9]),
        layer_topks=torch.tensor([1, 1]),
        min_avg_tokens_per_expert=10,
    )

    # Summing layers by rank first would incorrectly report 1.0.  Each
    # sequential layer actually waits for a rank carrying 200 units.
    assert result["critical_overhead_gflops_per_token"] == pytest.approx(0.75)
    assert result["prefill_ep_compute_critical_overhead_ratio"] == pytest.approx(1 / 3)


def test_balanced_layers_report_zero_overhead():
    stats = torch.zeros((3, 2, 4, 2), dtype=torch.int64)
    stats[:, :, :, 0] = 100
    stats[:, :, :, 1] = torch.tensor([128, 128, 128, 128])

    result = calculate_prefill_balance_stats(
        stats,
        total_routed_experts=8,
        layer_compute_flops=torch.tensor([1e9, 2e9]),
        layer_topks=torch.tensor([1, 1]),
        min_avg_tokens_per_expert=10,
    )

    assert result["critical_overhead_gflops_per_token"] == pytest.approx(0.0)
    assert result["prefill_ep_compute_critical_overhead_ratio"] == pytest.approx(0.0)


def test_sample_threshold_is_applied_to_the_complete_block():
    stats = torch.zeros((2, 1, 2, 2), dtype=torch.int64)
    stats[0, :, :, 0] = 1
    stats[0, :, :, 1] = torch.tensor([100, 100])
    stats[1, :, :, 0] = 100
    stats[1, :, :, 1] = torch.tensor([100, 200])

    result = calculate_prefill_balance_stats(
        stats,
        total_routed_experts=2,
        layer_compute_flops=torch.tensor([1e9]),
        layer_topks=torch.tensor([1]),
        min_avg_tokens_per_expert=10,
    )

    assert result["prefill_rounds"] == 2
    assert result["critical_overhead_gflops_per_token"] == pytest.approx(50 / 202)
    assert result["prefill_ep_compute_critical_overhead_ratio"] == pytest.approx(50 / 250)


def test_all_insufficient_prefill_rounds_return_none():
    stats = torch.ones((2, 1, 2, 2), dtype=torch.int64)

    result = calculate_prefill_balance_stats(
        stats,
        total_routed_experts=2,
        layer_compute_flops=torch.tensor([1e9]),
        layer_topks=torch.tensor([1]),
        min_avg_tokens_per_expert=10,
    )

    assert result is None


def test_zero_compute_load_returns_none_without_invalid_ratio():
    stats = torch.zeros((1, 1, 2, 2), dtype=torch.int64)
    stats[:, :, :, 0] = 100

    result = calculate_prefill_balance_stats(
        stats,
        total_routed_experts=2,
        layer_compute_flops=torch.tensor([1e9]),
        layer_topks=torch.tensor([1]),
        min_avg_tokens_per_expert=10,
    )

    assert result is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_accumulate_ep_balance():
    counts = torch.tensor([1, 128, 129, 0], dtype=torch.int32, device="cuda")
    counters = torch.zeros((2, 2), dtype=torch.int64, device="cuda")

    accumulate_ep_balance(counts, counters, is_prefill=True)
    accumulate_ep_balance(counts, counters, is_prefill=False)

    assert torch.equal(
        counters.cpu(),
        torch.tensor([[258, 512], [258, 258]], dtype=torch.int64),
    )
