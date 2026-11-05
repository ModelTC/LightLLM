"""检查计算分布评测的控制变量，避免把不同工作量误当作负载分布差异。"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def benchmark(monkeypatch):
    script = Path(__file__).resolve().parents[3] / "test/benchmark/kernel/benchmark_eplb_decode_compute.py"
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location("decode_compute_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def route_counts(routes):
    assert routes.shape == (128, 8)
    assert routes.min() >= 0 and routes.max() < 256
    assert torch.all(torch.sort(routes, dim=1).values[:, 1:] != torch.sort(routes, dim=1).values[:, :-1])
    return torch.bincount(routes.flatten().to(torch.long), minlength=256).reshape(8, 32)


@pytest.mark.parametrize("groups", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("bank", range(4))
def test_spread_changes_only_expert_fragmentation(benchmark, groups, bank):
    routes = benchmark.build_logical_routes(dict(pattern="spread", groups=groups), 128, bank)
    counts = route_counts(routes)
    assert torch.equal(counts.sum(dim=1), torch.full((8,), 128))
    assert torch.equal((counts > 0).sum(dim=1), torch.full((8,), groups))
    assert torch.equal(torch.sort(routes // 32, dim=1).values, torch.arange(8).expand(128, -1))
    assert int(counts.sum()) == 1024


@pytest.mark.parametrize("left", [4, 6, 7])
@pytest.mark.parametrize("bank", [0, 3])
@pytest.mark.parametrize("pattern", ["pair_compact", "pair_spread8"])
def test_pairs_keep_fanout_and_total_work_but_change_rank_balance(benchmark, pattern, left, bank):
    routes = benchmark.build_logical_routes(dict(pattern=pattern, left=left), 128, bank)
    counts = route_counts(routes)
    assert torch.equal(counts.sum(dim=1)[::2], torch.full((4,), 32 * left))
    assert torch.equal(counts.sum(dim=1)[1::2], torch.full((4,), 32 * (8 - left)))
    targets = torch.nn.functional.one_hot((routes // 32).to(torch.long), num_classes=8).any(dim=1)
    assert torch.equal(targets.sum(dim=1), torch.full((128,), 2))
    active = (counts > 0).sum(dim=1)
    if pattern == "pair_compact":
        assert torch.equal(active[::2], torch.full((4,), left))
        assert torch.equal(active[1::2], torch.full((4,), 8 - left))
    else:
        assert torch.equal(active, torch.full((8,), 8))
        assert torch.equal(((counts + 127) // 128 * 128).sum(dim=1), torch.full((8,), 1024))
    assert int(counts.sum()) == 1024


def test_one_target_has_same_compute_layout_as_spread8_but_less_communication(benchmark):
    routes = benchmark.build_logical_routes(dict(pattern="one_target"), 128, 0)
    counts = route_counts(routes)
    spread_counts = route_counts(benchmark.build_logical_routes(dict(pattern="spread", groups=8), 128, 0))
    assert torch.equal(counts, spread_counts)
    targets = torch.nn.functional.one_hot((routes // 32).to(torch.long), num_classes=8).any(dim=1)
    assert torch.equal(targets.sum(dim=1), torch.ones(128, dtype=torch.int64))


@pytest.mark.parametrize("pattern", ["uniform", "cooccurrence"])
def test_statistical_traffic_uses_distinct_topk_and_preserves_total_work(benchmark, pattern):
    routes = benchmark.build_logical_routes(dict(pattern=pattern), 128, 0)
    assert int(route_counts(routes).sum()) == 1024


def test_case_names_are_unique(benchmark):
    names = [case["name"] for case in benchmark.case_specs()]
    assert len(names) == len(set(names))


@pytest.fixture
def timing_samples():
    # W1、W2 的慢 rank 不同；不能把两个独立的 max 相加当作 GEMM 跨度。
    ranks = []
    for rank, phases in enumerate(([1, 1, 40, 2, 10, 1, 5], [1, 1, 15, 1, 35, 1, 6])):
        ranks.append(
            dict(
                phase_us=[[phases]],
                plain_us=[50],
                snapshots=[dict(counts=[1])],
                routes=[dict(expert_ids=[[0, 1]] if rank == 0 else [], remote_rank_sum=1 if rank == 0 else 0)],
                isolated_us=dict(w1=[30], w2=[20], compute=[45]),
                thermal_samples=[dict(max_temperature=50, thermal_slowdown=False, samples=[])],
            )
        )
    return ranks, SimpleNamespace(global_tokens=1, topk=2, banks=1)


def test_timing_share_uses_same_profiled_denominator_and_rankwise_gemm_sum(benchmark, timing_samples):
    ranks, args = timing_samples
    result = benchmark.summarize(dict(name="control"), ranks, args, 78, 118)
    assert result["critical_w1_us"] == 40
    assert result["critical_w2_us"] == 35
    assert result["critical_gemm_us"] == 50
    assert result["critical_compute_us"] == 52
    assert result["critical_gemm_share_pct"] == pytest.approx(100 * 50 / 60)
    assert result["critical_compute_share_pct"] == pytest.approx(100 * 52 / 60)
    assert result["plain_moe_us"] == 50


def test_received_counts_must_match_actual_routes(benchmark, timing_samples):
    ranks, args = timing_samples
    ranks[0]["snapshots"][0]["counts"] = [2]
    with pytest.raises(AssertionError):
        benchmark.summarize(dict(name="control"), ranks, args, 78, 118)


def test_any_rank_thermal_slowdown_is_visible_in_summary(benchmark, timing_samples):
    ranks, args = timing_samples
    ranks[1]["thermal_samples"][0].update(max_temperature=87, thermal_slowdown=True)
    result = benchmark.summarize(dict(name="control"), ranks, args, 78, 118)
    assert result["thermal_slowdown_observed"]
    assert result["max_measured_temperature"] == 87
