from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import lightllm.common.basemodel.cuda_graph as cuda_graph_module
from lightllm.common.basemodel.cuda_graph import CudaGraphRunner


@dataclass
class _GraphState:
    value: torch.Tensor

    def copy_for_cuda_graph(self, source):
        self.value.copy_(source.value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Graph")
def test_generic_cuda_graph_runner_replays_the_complete_capture(monkeypatch):
    monkeypatch.setattr(
        cuda_graph_module,
        "get_env_start_args",
        lambda: SimpleNamespace(enable_torch_memory_saver=False),
    )
    state = _GraphState(torch.tensor([2.0], device="cuda"))
    runner = CudaGraphRunner()
    assert runner.needs_capture("fixed")

    first = runner.run("fixed", state, lambda graph_state: graph_state.value * 3)
    torch.cuda.synchronize()
    assert not runner.needs_capture("fixed")
    assert runner.graph["fixed"][1] is state
    torch.testing.assert_close(first, torch.tensor([6.0], device="cuda"))

    state.value.fill_(4.0)
    second = runner.run("fixed", state, lambda graph_state: graph_state.value * 3)
    torch.cuda.synchronize()
    assert first.data_ptr() == second.data_ptr()
    torch.testing.assert_close(second, torch.tensor([12.0], device="cuda"))


@pytest.mark.parametrize("enabled", [False, True])
def test_generic_cuda_graph_runner_uses_shared_memory_saver_config(
    monkeypatch,
    enabled,
):
    wrapper = object()
    observed = []
    monkeypatch.setattr(
        cuda_graph_module,
        "get_env_start_args",
        lambda: SimpleNamespace(enable_torch_memory_saver=enabled),
    )
    monkeypatch.setattr(
        cuda_graph_module,
        "TorchMemorySaverWrapper",
        lambda configured: observed.append(configured) or wrapper,
    )

    runner = CudaGraphRunner(mempool=object())

    assert observed == [enabled]
    assert runner.torch_memory_saver is wrapper
