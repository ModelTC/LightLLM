from types import SimpleNamespace

import pytest

import lightllm.common.basemodel.cuda_graph as cuda_graph_module
from lightllm.common.basemodel.cuda_graph import CudaGraph


@pytest.fixture(autouse=True)
def _graph_args(monkeypatch):
    args = SimpleNamespace(
        enable_decode_microbatch_overlap=False,
        enable_tpsp_mix_mode=False,
        enable_torch_memory_saver=False,
    )
    monkeypatch.setattr(cuda_graph_module, "get_env_start_args", lambda: args)
    return args


def _batch_sizes(max_batch_size, batch_stride=1):
    physical_max_batch_size = max_batch_size * batch_stride
    graph = CudaGraph(
        batch_step_size_before_split=batch_stride,
        split_batch_size=4 * batch_stride,
        batch_step_size_after_split=2 * batch_stride,
        max_batch_size=physical_max_batch_size,
    )
    return graph.cuda_graph_batch_sizes


def test_dynamic_schedule_uses_compacted_physical_rows(_graph_args):
    assert _batch_sizes(max_batch_size=128) == [1, 2, 3, 4, *range(6, 129, 2)]


def test_public_static_schedule_preserves_original_static_mtp_default(_graph_args):
    assert CudaGraph.gen_cuda_graph_batch_sizes(
        batch_step_size_before_split=8,
        split_batch_size=32,
        batch_step_size_after_split=16,
        max_batch_size=32,
    ) == [
        8,
        16,
        24,
        32,
    ]


def test_instance_and_public_static_schedule_match(_graph_args):
    graph = CudaGraph(
        batch_step_size_before_split=8,
        split_batch_size=32,
        batch_step_size_after_split=16,
        max_batch_size=128,
    )

    assert graph.cuda_graph_batch_sizes == CudaGraph.gen_cuda_graph_batch_sizes(
        batch_step_size_before_split=8,
        split_batch_size=32,
        batch_step_size_after_split=16,
        max_batch_size=graph.max_batch_size,
        tp_world_size=graph.tp_world_size,
    )


def test_batch_step_size_before_split_controls_capture_range(_graph_args):
    assert _batch_sizes(max_batch_size=4, batch_stride=8) == [8, 16, 24, 32]


def test_batch_step_size_after_split_controls_capture_range(_graph_args):
    assert _batch_sizes(max_batch_size=8, batch_stride=7) == [
        7,
        14,
        21,
        28,
        42,
        56,
    ]


def test_extra_batch_sizes_are_merged_and_bounded(_graph_args):
    graph = CudaGraph(
        batch_step_size_before_split=6,
        split_batch_size=24,
        batch_step_size_after_split=96,
        max_batch_size=384,
        extra_batch_sizes=[1, 2, 4, 20, 36, 52, 64, 999],
    )

    assert graph.cuda_graph_batch_sizes == [
        1,
        2,
        4,
        6,
        12,
        18,
        20,
        24,
        36,
        52,
        64,
        120,
        216,
        312,
        384,
    ]
