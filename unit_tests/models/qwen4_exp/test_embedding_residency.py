import argparse

import pytest
import torch

from lightllm.models.qwen4_exp.layer_weights import embedding_weight as module
from lightllm.server.api_cli import add_cli_args
from lightllm.server.core.objs.start_args_type import StartArgs


def test_residency_is_opt_in_and_cli_matches_start_args():
    parser = add_cli_args(argparse.ArgumentParser())
    assert not parser.parse_args([]).offload_ngram_embedding
    assert not StartArgs().offload_ngram_embedding
    assert parser.parse_args(["--offload_ngram_embedding"]).offload_ngram_embedding


@pytest.mark.parametrize("args", [{}, {"offload_ngram_embedding": False}])
def test_gpu_factory_preserves_existing_constructor(args, monkeypatch):
    monkeypatch.setattr(module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(module, "RowShardedEmbeddingWeight", lambda *values: values)
    result = module.ngram_embedding_weight(160, 1024, "table.weight", torch.bfloat16, 4)
    assert result == (160, 1024, "table.weight", torch.bfloat16, 4)


@pytest.mark.parametrize("torch_fallback", [False, True])
def test_cpu_factory_injects_model_owned_allocation_policy(monkeypatch, torch_fallback):
    monkeypatch.setattr(
        module,
        "get_env_start_args",
        lambda: {"offload_ngram_embedding": True, "hardware_platform": "cuda", "enable_torch_fallback": torch_fallback},
    )
    monkeypatch.setattr(module, "PinnedRowShardedEmbeddingWeight", lambda *values, **kwargs: (values, kwargs))
    from lightllm.models.qwen4_exp.pinned_memory import pinned_zeros

    values, kwargs = module.ngram_embedding_weight(160, 1024, "table.weight", torch.bfloat16, 4)
    assert values == (160, 1024, "table.weight", torch.bfloat16, 4)
    assert kwargs == {"cpu_allocator": pinned_zeros}


@pytest.mark.parametrize("extra", [{"hardware_platform": "musa"}, {"hardware_platform": "cpu"}])
def test_unsupported_offload_configuration_fails_before_allocation(extra, monkeypatch):
    monkeypatch.setattr(module, "get_env_start_args", lambda: dict(offload_ngram_embedding=True, **extra))
    with pytest.raises(ValueError, match="CUDA Triton"):
        module.ngram_embedding_weight(160, 1024, "table.weight", torch.bfloat16, 4)
