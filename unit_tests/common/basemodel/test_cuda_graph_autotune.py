from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel import cuda_graph as cuda_graph_module
from lightllm.common.basemodel.cuda_graph import CudaGraph
from lightllm.common.triton_utils import autotuner as autotuner_module
from lightllm.common.triton_utils.autotuner import AutotuneKernelType, AutotuneLevel, Autotuner, autotune


@pytest.fixture
def capture_env(monkeypatch):
    env = SimpleNamespace(capturing=False, capture_count=0, graphs=[])
    env.args = SimpleNamespace(
        enable_decode_microbatch_overlap=False,
        enable_tpsp_mix_mode=False,
        enable_torch_memory_saver=False,
    )
    monkeypatch.setattr(cuda_graph_module, "get_env_start_args", lambda: env.args)
    monkeypatch.setattr(Autotuner, "_autotune_warmup_kernel_type", None)
    monkeypatch.setattr(autotuner_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(autotuner_module.KernelConfigs, "get_config_file_name", lambda params: "configs.json")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)

    class FakeGraph:
        def __init__(self):
            self.replays = 0
            env.graphs.append(self)

        def replay(self):
            self.replays += 1

    @contextmanager
    def capture(graph, **kwargs):
        assert not Autotuner.is_autotune_warmup()
        env.capture_count += 1
        env.capturing = True
        try:
            yield
        finally:
            env.capturing = False

    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", capture)
    return env


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_decode_attention_tunes_before_capture_only(capture_env, tmp_path, monkeypatch, overlap, level):
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: level)
    capture_env.args.enable_decode_microbatch_overlap = overlap
    graph = CudaGraph(1, 2, 1, max_batch_size=2)
    benchmarks = []

    def make_kernel(kernel_type):
        @autotune(
            kernel_name=kernel_type.value,
            kernel_type=kernel_type,
            configs_gen_func=lambda: [{"block": 1}, {"block": 2}],
            static_key_func=lambda: {},
            run_key_func=lambda size: size,
        )
        def kernel(size, run_config=None):
            return size

        def bench(size, run_config):
            assert kernel_type == AutotuneKernelType.DECODE_ATTENTION
            assert Autotuner.is_kernel_autotune_warmup(kernel_type)
            assert not capture_env.capturing
            benchmarks.append(run_config)
            return 1.0 / run_config["block"]

        cache_dir = tmp_path / kernel_type.value
        cache_dir.mkdir()
        kernel._cache_dir = str(cache_dir)
        monkeypatch.setattr(kernel, "_bench", bench)
        return kernel, cache_dir / "configs.json"

    general, general_cache = make_kernel(AutotuneKernelType.GENERAL)
    attention, attention_cache = make_kernel(AutotuneKernelType.DECODE_ATTENTION)
    states = [SimpleNamespace(input_ids=torch.ones(2, dtype=torch.int64)) for _ in range(2 if overlap else 1)]
    outputs = [object() for _ in states]
    forward_calls = []

    def decode_func(*infer_states):
        forward_calls.append(capture_env.capturing)
        assert Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.DECODE_ATTENTION) == (not capture_env.capturing)
        for original, state in zip(states, infer_states):
            assert (state is original) == capture_env.capturing
            assert not hasattr(state, "temporary_buffer")
            state.temporary_buffer = object()
            general(state.input_ids.shape[0])
            attention(state.input_ids.shape[0])
        return tuple(outputs) if overlap else outputs[0]

    result = graph.capture_decode(decode_func, *states)
    assert result == (tuple(outputs) if overlap else outputs[0])
    assert forward_calls == [False, True]
    assert not Autotuner.is_autotune_warmup()
    assert capture_env.capture_count == 1
    assert capture_env.graphs[0].replays == 1
    expected_benchmarks = 0
    if level in [AutotuneLevel.ADAPTIVE_AUTOTUNE, AutotuneLevel.FORCE_AUTOTUNE]:
        expected_benchmarks = 2
    assert len(benchmarks) == expected_benchmarks
    assert attention_cache.exists() == (expected_benchmarks > 0)
    assert not general_cache.exists()


@pytest.mark.parametrize("overlap", [False, True])
def test_warmup_failure_restores_phase_without_capturing(capture_env, overlap):
    capture_env.args.enable_decode_microbatch_overlap = overlap
    graph = CudaGraph(1, 2, 1, max_batch_size=2)
    states = [SimpleNamespace(input_ids=torch.ones(2, dtype=torch.int64)) for _ in range(2 if overlap else 1)]

    def decode_func(*infer_states):
        assert Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.DECODE_ATTENTION)
        raise RuntimeError("decode warmup failed")

    with Autotuner.autotune_warmup(AutotuneKernelType.GENERAL):
        with pytest.raises(RuntimeError, match="decode warmup failed"):
            graph.capture_decode(decode_func, *states)
        assert Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.GENERAL)
    assert not Autotuner.is_autotune_warmup()
    assert capture_env.capture_count == 0
    assert graph.graph == {}
