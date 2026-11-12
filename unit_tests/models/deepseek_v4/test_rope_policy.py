from types import SimpleNamespace

import pytest
import torch

from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel
from lightllm.models.deepseek_v4.triton_kernel.norm_rope import gather_rope_tables


@pytest.fixture(params=["type", "rope_type"])
def model(monkeypatch, request):
    arange = torch.arange

    def cpu_arange(*args, **kwargs):
        kwargs["device"] = "cpu"
        return arange(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", cpu_arange)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    layers = [
        SimpleNamespace(compress_ratio=ratio, index_infer=SimpleNamespace() if ratio == 4 else None)
        for ratio in (0, 4, 128)
    ]
    model = DeepseekV4TpPartModel.__new__(DeepseekV4TpPartModel)
    model.config = {
        "qk_rope_head_dim": 64,
        "max_position_embeddings": 65537,
        "rope_theta": 10000,
        "compress_rope_theta": 160000,
        "rope_scaling": {
            request.param: "yarn",
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "beta_fast": 32,
            "beta_slow": 1,
        },
    }
    model.max_seq_length = 8192
    model.layers_infer = layers
    model._init_to_get_rotary()
    return model


def test_pure_swa_uses_plain_rope_even_with_yarn_config(model):
    positions = torch.tensor([0, 1, 4096, 65535, 65536])
    inv_freq = 1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    angles = torch.outer(positions.float(), inv_freq)
    expected = torch.polar(torch.ones_like(angles), angles)

    torch.testing.assert_close(model.layers_infer[0].freqs_cis[positions], expected, rtol=1e-6, atol=1e-6)
    assert model._freqs_cis_sliding.dtype == torch.complex64
    assert model._cos_cached_sliding.dtype == torch.float32


def test_compressed_layers_keep_yarn_for_local_attention_and_indexer(model):
    # The lowest-frequency pair is fully interpolated by YaRN (factor 16).
    positions = torch.tensor([0, 1, 4096, 65535, 65536])
    inv_freq = 1.0 / (160000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    angles = positions.float() * (inv_freq[-1] / 16)
    expected = torch.polar(torch.ones_like(angles), angles)

    for layer in model.layers_infer[1:]:
        torch.testing.assert_close(layer.freqs_cis[positions, -1], expected, rtol=1e-6, atol=1e-6)
        assert layer.freqs_cis is model._freqs_cis_compress
        assert layer.cos_compress_table is model._cos_cached_compress
        assert layer.sin_compress_table is model._sin_cached_compress
    assert model.layers_infer[1].index_infer.freqs_cis is model._freqs_cis_compress


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", [0, 1, 385])
def test_gather_rope_tables(rows):
    sliding = torch.randn((512, 32), dtype=torch.complex64, device="cuda")
    compress = torch.randn_like(sliding)
    tables = (sliding.real, sliding.imag, compress.real.contiguous(), compress.imag)
    positions = torch.arange(rows * 2, dtype=torch.int32, device="cuda").remainder_(512)[::2]

    def check(outputs):
        for output, table in zip(outputs, tables):
            assert output.is_contiguous()
            torch.testing.assert_close(output, table.index_select(0, positions), atol=0, rtol=0)

    check(gather_rope_tables(positions, tables))
    if rows:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = gather_rope_tables(positions, tables)
        positions.add_(3).remainder_(512)
        graph.replay()
        check(captured)
