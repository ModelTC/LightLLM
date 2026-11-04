from types import SimpleNamespace
import subprocess
import sys

import pytest
import torch

from lightllm.core.runner.graph import GraphRunner
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager


@pytest.mark.parametrize("module", ["decode_graph", "prefill_graph"])
def test_graph_module_can_be_imported_before_the_model(module):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import lightllm.core.runner.{module}; from lightllm.common.basemodel import TpPartBaseModel",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_eager_prefill_failure_releases_temporary_tensor_cache():
    runner = GraphRunner()
    batch = SimpleNamespace(input_ids=torch.ones(4))

    def fail(inputs, batch):
        g_cache_manager.alloc_tensor((4,), torch.float32, device="cpu")
        raise ValueError("injected model failure")

    with pytest.raises(ValueError, match="injected model failure"):
        runner.run_prefill(fail, [], batch)
    assert not g_cache_manager.cache_env_ok
    assert not g_cache_manager.ptr_to_bufnode
    expected = torch.ones(4)
    assert runner.run_prefill(lambda inputs, batch: inputs, [expected], batch)[0] is expected
    assert not g_cache_manager.cache_env_ok


def test_close_resets_decode_and_prefill_segments_before_releasing_bindings():
    from lightllm.core.runner.decode_graph import CudaGraph
    from lightllm.core.runner.prefill_graph import PrefillCudaGraph
    from lightllm.core.runner.graph_program import GraphProgram

    resets = []
    runner = GraphRunner()
    runner.decode = CudaGraph.__new__(CudaGraph)
    runner.decode.graph = {1: (SimpleNamespace(reset=lambda: resets.append("decode")), object(), object())}
    program = GraphProgram(pool=object())
    program.steps = [SimpleNamespace(reset=lambda: resets.append("prefill")), lambda batch: None]
    program._complete = True
    runner.prefill = PrefillCudaGraph.__new__(PrefillCudaGraph)
    runner.prefill.graph = {4: (SimpleNamespace(prefill_program=program), [], [], None)}
    decode, prefill = runner.decode, runner.prefill
    runner.close()
    runner.close()
    assert resets == ["prefill", "decode"]
    assert not decode.graph and not prefill.graph and not program.steps
    assert runner.decode is runner.prefill is decode.mempool is prefill.mempool is program.pool is None
    with pytest.raises(RuntimeError, match="incomplete"):
        program.replay(None)
