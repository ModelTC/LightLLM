"""评测保护与小 batch 行号重置的 CPU 测试，不占用 GPU。"""

import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def benchmark_modules(monkeypatch):
    benchmark_path = Path(__file__).resolve().parents[3] / "test/benchmark/kernel"
    monkeypatch.syspath_prepend(str(benchmark_path))
    return (
        importlib.import_module("benchmark_eplb_decode_rank_weighted"),
        importlib.import_module("benchmark_eplb_decode_rank_weighted_moe"),
    )


@pytest.mark.parametrize("conflict", ["none", "visible_process", "hidden_container", "other_rank", "snapshot_lag"])
def test_benchmark_rejects_external_gpu_occupancy(benchmark_modules, monkeypatch, conflict):
    _, benchmark = benchmark_modules
    mib = 1024 ** 2
    processes = [SimpleNamespace(pid=os.getpid(), usedGpuMemory=100 * mib)]
    if conflict == "visible_process":
        processes.append(SimpleNamespace(pid=os.getpid() + 1, usedGpuMemory=20 * mib))
    used = sum(process.usedGpuMemory for process in processes) + (66000 if conflict == "hidden_container" else 5) * mib
    snapshots = iter([used + 2048 * mib, used, used] if conflict == "snapshot_lag" else [used, used, used])
    nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByUUID=lambda uuid: uuid,
        nvmlDeviceGetComputeRunningProcesses=lambda handle: processes,
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=next(snapshots, used)),
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr(benchmark.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(benchmark.torch.cuda, "get_device_properties", lambda rank: SimpleNamespace(uuid="GPU-test"))
    monkeypatch.setattr(benchmark.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(benchmark.dist, "get_world_size", lambda: 2)
    group = object()

    def gather(snapshots, local, group):
        snapshots[:] = [local, dict(rank=1, other_pids=[], unaccounted_mib=2048 if conflict == "other_rank" else 0)]

    monkeypatch.setattr(benchmark.dist, "all_gather_object", gather)
    if conflict in ("none", "snapshot_lag"):
        benchmark.check_gpu_availability(control_group=group)
    else:
        with pytest.raises(RuntimeError, match="GPU 被其他任务占用"):
            benchmark.check_gpu_availability(control_group=group)


def test_cpu_metrics_reset_token_indices_for_each_real_batch(benchmark_modules):
    benchmark, _ = benchmark_modules
    placement = [[0, 1], [0, 2], [1, 3], [2, 3]]
    logical = np.tile(np.asarray([[0, 1, 2, 3]], dtype=np.int32), (16, 1))
    common = dict(logical=logical, placement=placement, source_rank=3, node_world_size=4, variant="current")
    repeated = benchmark.route_cpu(**common, token_period=4)
    for start in (4, 8, 12):
        assert np.array_equal(repeated[:4], repeated[start : start + 4])
    assert not np.array_equal(repeated, benchmark.route_cpu(**common))
    # 本卡专家 2/3 始终固定，不能被重置行号或随机评分移走。
    assert np.all(repeated[:, 2:] == np.asarray([6, 7]))


def test_weighted_cpu_reference_does_not_depend_on_token_row_indices(benchmark_modules):
    benchmark, _ = benchmark_modules
    placement = [[0, 1], [0, 2], [1, 3], [2, 3]]
    logical = np.tile(np.asarray([[0, 1, 2, 3]], dtype=np.int32), (16, 1))
    common = dict(logical=logical, placement=placement, source_rank=3, node_world_size=4, variant="weighted")
    periodic = benchmark.route_cpu(**common, token_period=4)
    np.testing.assert_array_equal(periodic, benchmark.route_cpu(**common))
    np.testing.assert_array_equal(periodic, np.repeat(periodic[:1], len(logical), axis=0))


def test_benchmarks_only_expose_current_and_weighted_routes(benchmark_modules):
    benchmark, moe = benchmark_modules
    assert set(moe.VARIANTS) == {"current", "weighted"}
    with pytest.raises(ValueError, match="Unsupported routing variant"):
        benchmark.route_cpu(
            logical=np.asarray([[0, 1]], dtype=np.int32),
            placement=[[0, 1], [0, 1]],
            source_rank=0,
            node_world_size=2,
            variant="greedy",
        )


@pytest.mark.parametrize("tied_slot", [4, 7])
def test_weighted_benchmark_accepts_each_maximum_score_replica(benchmark_modules, tied_slot):
    benchmark, _ = benchmark_modules
    benchmark.validate_route_cpu(
        logical=np.asarray([[1, 5, 7, 0]], dtype=np.int32),
        output=np.asarray([[tied_slot, 5, 8, 9]], dtype=np.int32),
        placement=[[2, 3, 4], [0, 1, 5], [0, 1, 7], [0, 6, 2]],
        source_rank=3,
        node_world_size=4,
        variant="weighted",
    )


@pytest.mark.parametrize("invalid_output", [[0, 3, 7], [4, 6, 7]])
def test_weighted_benchmark_rejects_lower_score_and_nonlocal_replicas(benchmark_modules, invalid_output):
    benchmark, _ = benchmark_modules
    # expert 1 在 rank 1 的槽 3 可执行，但 rank 2 覆盖 1/2，分数更高。
    # expert 0 的 rank 1 槽 4 可执行，但本卡已有槽 0，不能绕过本卡优先。
    with pytest.raises(AssertionError):
        benchmark.validate_route_cpu(
            logical=np.asarray([[0, 1, 2]], dtype=np.int32),
            output=np.asarray([invalid_output], dtype=np.int32),
            placement=[[0, 3, 4], [1, 0, 5], [1, 2, 6], [3, 4, 7]],
            source_rank=0,
            node_world_size=4,
            variant="weighted",
        )


@pytest.mark.parametrize("normalize_replica_coverage,expected_slot", [(False, 5), (True, 10)])
def test_weighted_cpu_reference_supports_inverse_replica_scores(
    benchmark_modules, normalize_replica_coverage, expected_slot
):
    benchmark, _ = benchmark_modules
    placement = [[4, 5, 6, 7, 8], [0, 1, 2, 4, 9], [0, 3, 5, 6, 7], [0, 1, 2, 8, 9], [1, 2, 4, 5, 9]]
    logical = np.asarray([[0, 1, 2, 3, 4]], dtype=np.int32)
    output = benchmark.route_cpu(
        logical=logical,
        placement=placement,
        source_rank=0,
        node_world_size=5,
        variant="weighted",
        normalize_replica_coverage=normalize_replica_coverage,
    )
    assert output[0, 0] == expected_slot
    benchmark.validate_route_cpu(
        logical=logical,
        output=output,
        placement=placement,
        source_rank=0,
        node_world_size=5,
        variant="weighted",
        normalize_replica_coverage=normalize_replica_coverage,
    )
    # 同一个候选槽在两种评分下的合法性不同，校验器不能只验证物理 ID 对应同一逻辑专家。
    with pytest.raises(AssertionError):
        benchmark.validate_route_cpu(
            logical=logical,
            output=output,
            placement=placement,
            source_rank=0,
            node_world_size=5,
            variant="weighted",
            normalize_replica_coverage=not normalize_replica_coverage,
        )


@pytest.mark.parametrize("source_rank", [0, 3, 7])
def test_cpu_metrics_use_same_philox_stream_as_independent_reference(benchmark_modules, source_rank):
    from unit_tests.common.fused_moe.test_eplb_decode_random import token_random_reference

    benchmark, _ = benchmark_modules
    tokens = np.asarray([0, 1, 127, 1 << 31, (1 << 32) - 1], dtype=np.uint32)
    expected = np.asarray([token_random_reference(source_rank, int(token)) for token in tokens], dtype=np.uint32)
    np.testing.assert_array_equal(
        benchmark.token_random_values(source_rank=source_rank, token_indices=tokens), expected
    )
