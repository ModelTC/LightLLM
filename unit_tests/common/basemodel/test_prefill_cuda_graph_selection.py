from types import SimpleNamespace

import pytest

import lightllm.common.basemodel.prefill_cuda_graph as prefill_cuda_graph


class _DecodeGraph:
    mempool = object()


def _args(**overrides):
    values = {
        "enable_prefill_microbatch_overlap": False,
        "prefill_cudagraph_max_handle_token": 32768,
        "prefill_cudagraph_token_nums": None,
        "prefill_cudagraph_batch_sizes": None,
        "prefill_cudagraph_capture_attention": False,
        "batch_max_tokens": 32768,
        "enable_tpsp_mix_mode": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_explicit_token_nums_only_run_exact_shapes(monkeypatch):
    monkeypatch.setattr(
        prefill_cuda_graph,
        "get_env_start_args",
        lambda: _args(
            prefill_cudagraph_token_nums=[17152, 352, 17152],
            prefill_cudagraph_batch_sizes=[64, 1, 64],
        ),
    )

    graph = prefill_cuda_graph.PrefillCudaGraph(_DecodeGraph(), tp_world_size=8)

    assert graph.graph_handle_token_nums == [352, 17152]
    assert graph.can_run(352, batch_size=1, max_q_seq_len=352, max_kv_seq_len=352, max_cache_len=0)
    assert graph.can_run(17152, batch_size=64, max_q_seq_len=268, max_kv_seq_len=268, max_cache_len=0)
    assert not graph.can_run(17152, batch_size=1, max_q_seq_len=17152, max_kv_seq_len=17152, max_cache_len=0)
    assert not graph.can_run(17152, batch_size=64, max_q_seq_len=269, max_kv_seq_len=269, max_cache_len=0)
    assert not graph.can_run(17152, batch_size=64, max_q_seq_len=268, max_kv_seq_len=300, max_cache_len=32)
    assert not graph.can_run(351, batch_size=1, max_q_seq_len=351, max_kv_seq_len=351, max_cache_len=0)
    assert not graph.can_run(16000, batch_size=64, max_q_seq_len=250, max_kv_seq_len=250, max_cache_len=0)


def test_explicit_token_nums_respect_limits(monkeypatch):
    monkeypatch.setattr(
        prefill_cuda_graph,
        "get_env_start_args",
        lambda: _args(
            prefill_cudagraph_max_handle_token=20000,
            batch_max_tokens=18000,
            prefill_cudagraph_token_nums=[0, 17152, 20000],
            prefill_cudagraph_batch_sizes=[1, 64, 64],
        ),
    )

    graph = prefill_cuda_graph.PrefillCudaGraph(_DecodeGraph(), tp_world_size=8)

    assert graph.graph_handle_token_nums == [17152]


def test_explicit_token_nums_reject_empty_valid_set(monkeypatch):
    monkeypatch.setattr(
        prefill_cuda_graph,
        "get_env_start_args",
        lambda: _args(
            prefill_cudagraph_token_nums=[0, 40000],
            prefill_cudagraph_batch_sizes=[1, 64],
        ),
    )

    with pytest.raises(ValueError, match="prefill_cudagraph_token_nums"):
        prefill_cuda_graph.PrefillCudaGraph(_DecodeGraph(), tp_world_size=8)


def test_explicit_layout_rejects_missing_or_mismatched_batch_sizes(monkeypatch):
    monkeypatch.setattr(
        prefill_cuda_graph,
        "get_env_start_args",
        lambda: _args(prefill_cudagraph_token_nums=[17152]),
    )
    with pytest.raises(ValueError, match="prefill_cudagraph_batch_sizes is required"):
        prefill_cuda_graph.PrefillCudaGraph(_DecodeGraph(), tp_world_size=8)

    monkeypatch.setattr(
        prefill_cuda_graph,
        "get_env_start_args",
        lambda: _args(
            prefill_cudagraph_token_nums=[17152, 352],
            prefill_cudagraph_batch_sizes=[64],
        ),
    )
    with pytest.raises(ValueError, match="same number of entries"):
        prefill_cuda_graph.PrefillCudaGraph(_DecodeGraph(), tp_world_size=8)


def test_explicit_layout_rejects_nonuniform_sequence_shape(monkeypatch):
    monkeypatch.setattr(
        prefill_cuda_graph,
        "get_env_start_args",
        lambda: _args(
            prefill_cudagraph_token_nums=[17153],
            prefill_cudagraph_batch_sizes=[64],
        ),
    )
    with pytest.raises(ValueError, match="must be divisible"):
        prefill_cuda_graph.PrefillCudaGraph(_DecodeGraph(), tp_world_size=8)


def test_attention_capture_requires_exact_layout(monkeypatch):
    monkeypatch.setattr(
        prefill_cuda_graph,
        "get_env_start_args",
        lambda: _args(prefill_cudagraph_capture_attention=True),
    )

    with pytest.raises(ValueError, match="requires --prefill_cudagraph_token_nums"):
        prefill_cuda_graph.PrefillCudaGraph(_DecodeGraph(), tp_world_size=8)
