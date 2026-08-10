from types import SimpleNamespace

import pytest

import lightllm.common.basemodel.cuda_graph as cuda_graph_module
from lightllm.common.basemodel.cuda_graph import CudaGraph


@pytest.fixture(autouse=True)
def _graph_args(monkeypatch):
    args = SimpleNamespace(
        graph_split_batch_size=4,
        graph_grow_step_size=2,
        enable_decode_microbatch_overlap=False,
        enable_tpsp_mix_mode=False,
        enable_torch_memory_saver=False,
    )
    monkeypatch.setattr(cuda_graph_module, "get_env_start_args", lambda: args)
    return args


def _batch_sizes(max_batch_size, batch_multiplier=1):
    physical_max_batch_size = max_batch_size * batch_multiplier
    graph = CudaGraph(
        max_batch_size=physical_max_batch_size,
        batch_multiplier=batch_multiplier,
    )
    return graph.cuda_graph_batch_sizes


def test_dynamic_schedule_uses_compacted_physical_rows(_graph_args):
    assert _batch_sizes(max_batch_size=128) == [1, 2, 3, 4, *range(6, 129, 2)]


def test_public_static_schedule_preserves_original_static_mtp_default(_graph_args):
    assert CudaGraph.gen_cuda_graph_batch_sizes(max_batch_size=32, batch_multiplier=8) == [
        8,
        16,
        24,
        32,
    ]


def test_instance_and_public_static_schedule_match(_graph_args):
    graph = CudaGraph(max_batch_size=128, batch_multiplier=8)

    assert graph.cuda_graph_batch_sizes == CudaGraph.gen_cuda_graph_batch_sizes(
        max_batch_size=graph.max_batch_size,
        tp_world_size=graph.tp_world_size,
        batch_multiplier=8,
    )


def test_legacy_vanilla_layout_can_keep_k_plus_one_stride(_graph_args):
    assert _batch_sizes(max_batch_size=4, batch_multiplier=8) == [
        8,
        16,
        24,
        32,
    ]


def test_block_draft_layout_only_pads_to_complete_physical_blocks(_graph_args):
    assert _batch_sizes(max_batch_size=8, batch_multiplier=7) == [
        7,
        14,
        21,
        28,
        42,
        56,
    ]
