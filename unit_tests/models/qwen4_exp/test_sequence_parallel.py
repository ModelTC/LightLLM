from types import SimpleNamespace

import pytest
import torch

from lightllm.models.qwen3_5.layer_infer.transformer_layer_infer import Qwen35TransformerLayerInfer
from lightllm.models.qwen4_exp.layer_infer import transformer_layer_infer as decoder


def partitioned_layer(world, rank, enabled, full):
    layer = decoder.Qwen4ExpTransformerLayerInfer.__new__(decoder.Qwen4ExpTransformerLayerInfer)
    layer.tp_world_size_, layer.tp_rank_ = world, rank
    layer.layer_num_, layer.is_linear_attention_layer = 1, True
    layer._tpsp_allgather = lambda value, state: full if enabled else value
    layer._tpsp_sp_split = lambda value, state: value.chunk(world)[rank].contiguous() if enabled else value
    return layer


@pytest.mark.parametrize("world,rank", [(2, 0), (2, 1), (4, 0), (4, 3)])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("method", ["context_attention_forward", "token_attention_forward"])
def test_gdn_uses_global_rows_and_returns_local_partition(monkeypatch, world, rank, enabled, method):
    full = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    layer = partitioned_layer(world, rank, enabled, full)
    monkeypatch.setattr(decoder, "get_env_start_args", lambda: SimpleNamespace(enable_tpsp_mix_mode=enabled))

    def ordered(value, group, size, scatter_rank=None):
        total = value * size
        return total if scatter_rank is None else total.chunk(size)[scatter_rank].contiguous()

    def gdn(self, value, state, weight):
        assert torch.equal(value, full)
        return self._reduce_linear_output(value, state)

    monkeypatch.setattr(decoder, "ordered_tp_sum", ordered)
    monkeypatch.setattr(Qwen35TransformerLayerInfer, method, gdn)
    value = full.chunk(world)[rank].contiguous() if enabled else full
    actual = getattr(layer, method)(value, SimpleNamespace(dist_group=None), None)
    expected = (full * world).chunk(world)[rank] if enabled else full * world
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("world,rank", [(2, 0), (2, 1), (4, 0), (4, 3)])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("commit", [None, False, True])
def test_ple_keeps_global_metadata_and_local_residual(world, rank, enabled, prefill, commit):
    full = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    lexical = full / 16
    layer = partitioned_layer(world, rank, enabled, full)
    layer.context_attention_forward = layer.token_attention_forward = lambda value, *args: torch.zeros_like(value)
    layer._ffn = lambda value, *args: torch.zeros_like(value)
    tokens, metadata, batch = torch.arange(8), object(), SimpleNamespace()
    calls = []

    def apply(value, ids, weight, size, group):
        assert torch.equal(value, full) and ids is tokens and size == world
        return lexical

    def record(kind, observed_metadata, observed_batch):
        assert observed_metadata is metadata and observed_batch is batch
        calls.append(kind)

    batch.apply = apply
    cache = SimpleNamespace(commit=lambda *args: record("commit", *args))
    state = SimpleNamespace(
        qwen4_exp_ple={1: batch},
        input_ids=tokens,
        qwen4_exp_metadata=metadata,
        qwen4_exp_auto_commit=commit,
        dist_group=None,
        qwen4_exp_lexical_state=SimpleNamespace(
            lexical_caches={1: cache},
            stage_speculative_lexical=lambda number, *args: record("stage", *args),
        ),
    )
    residual = SimpleNamespace(read=lambda value: (value, None), write=lambda value, update, gates: value + update)
    weight = SimpleNamespace(
        ple=True,
        attn_hyper_connection=SimpleNamespace(component=residual),
        mlp_hyper_connection=SimpleNamespace(component=residual),
    )
    value = full.chunk(world)[rank].contiguous() if enabled else full
    actual = layer._forward(value, state, weight, prefill)
    expected = (full + lexical).chunk(world)[rank] if enabled else full + lexical
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert calls == ([] if commit is None else ["commit" if commit else "stage"])
