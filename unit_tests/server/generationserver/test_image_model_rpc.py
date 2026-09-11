from types import SimpleNamespace

import pytest
import torch

import lightllm.server.generationserver.prefix_attention as attention_module
from lightllm.server.activation_transfer import KVActivation
from lightllm.server.generationserver.kv_materializer import (
    DensePrefixKV,
    DensePrefixKVLayer,
    gather_dense_prefix_kv,
)
from lightllm.server.generationserver.prefix_attention import (
    DensePrefixAttentionProvider,
)


def _dense_prefix(key, value, *, layer_count=5):
    return DensePrefixKV(
        layers=tuple(
            DensePrefixKVLayer(key.clone(), value.clone())
            for _ in range(layer_count)
        ),
        page_table=torch.arange(key.shape[0], dtype=torch.int32).view(1, -1),
        prefix_lengths=torch.tensor([key.shape[0]], dtype=torch.int32),
        position_offsets=torch.tensor([key.shape[0]], dtype=torch.int32),
    )


def test_dense_provider_forwards_layer_query_and_stable_q_lengths(monkeypatch):
    calls = []

    def attend(*args, **kwargs):
        calls.append((args, kwargs))
        q = args[0]
        return torch.zeros_like(q), torch.zeros(q.shape[:2], dtype=torch.float32)

    monkeypatch.setattr(attention_module, "_paged_prefix_attention", attend)
    key = torch.randn(11, 1, 8)
    provider = DensePrefixAttentionProvider(_dense_prefix(key, key))
    query = torch.randn(5, 4, 8)

    output, lse = provider.attend(layer_index=3, q=query, softmax_scale=8**-0.5)
    assert provider.seq_len == 11
    assert output.shape == query.shape and lse.shape == query.shape[:2]
    assert calls[0][0][1].shape == key.shape
    assert calls[0][0][2].shape == key.shape
    torch.testing.assert_close(calls[0][0][5], torch.tensor([0, 5], dtype=torch.int32))
    assert calls[0][1]["max_q_len"] == 5

    provider.attend(layer_index=4, q=query, softmax_scale=8**-0.5)
    assert calls[0][0][5] is calls[1][0][5]


def test_dense_provider_rejects_a_backend_scale_mismatch():
    key = torch.zeros((11, 1, 8))
    provider = DensePrefixAttentionProvider(_dense_prefix(key, key, layer_count=1))

    with pytest.raises(ValueError, match="scales differ"):
        provider.attend(layer_index=0, q=torch.randn(2, 1, 8), softmax_scale=1.0)


def test_gather_dense_prefix_owns_a_copy_of_paged_kv():
    key = torch.arange(16 * 2 * 4, dtype=torch.float32).view(16, 2, 4)
    value = key + 1000
    page_table = torch.tensor(
        [[0, 1, 2, 3, 4], [11, 3, 13, 5, 9]],
        dtype=torch.int32,
    )
    memory = SimpleNamespace(
        layer_num=2,
        get_att_input_params=lambda _layer: (key, value),
    )
    dense = gather_dense_prefix_kv(
        memory,
        page_table,
        KVActivation(7, 19, 1, 5),
    )

    expected_indexes = page_table[1].long()
    torch.testing.assert_close(dense.layer(0).key, key[expected_indexes])
    torch.testing.assert_close(dense.layer(1).value, value[expected_indexes])
    key.zero_()
    assert torch.count_nonzero(dense.layer(0).key) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_triton_gather_packs_all_layers_and_ragged_requests():
    torch.manual_seed(31)
    layer_count, pool_tokens, heads, head_dim = 3, 23, 2, 16
    kv_buffer = torch.randn(
        layer_count,
        pool_tokens,
        2 * heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    request_table = torch.tensor(
        [[7, 2, 11, 0, 0], [5, 13, 3, 9, 1]],
        dtype=torch.int32,
        device="cuda",
    )
    memory = SimpleNamespace(
        layer_num=layer_count,
        head_num=heads,
        kv_buffer=kv_buffer,
        get_att_input_params=lambda layer: (
            kv_buffer[layer, :, :heads],
            kv_buffer[layer, :, heads:],
        ),
    )
    dense = gather_dense_prefix_kv(
        memory,
        request_table,
        (
            KVActivation(3, 31, 0, 3),
            KVActivation(4, 32, 1, 5),
        ),
    )
    indexes = torch.tensor(
        [7, 2, 11, 5, 13, 3, 9, 1], dtype=torch.long, device="cuda"
    )

    for layer_index in range(layer_count):
        key = dense.layer(layer_index).key
        value = dense.layer(layer_index).value
        assert key.is_contiguous() and value.is_contiguous()
        torch.testing.assert_close(key, kv_buffer[layer_index, indexes, :heads])
        torch.testing.assert_close(value, kv_buffer[layer_index, indexes, heads:])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA Prefix attention"
)
def test_bf16_dense_provider_matches_reference_attention():
    torch.manual_seed(7)
    key = torch.randn(5, 2, 64, dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    dense = DensePrefixKV(
        layers=(DensePrefixKVLayer(key, value),),
        page_table=torch.arange(5, dtype=torch.int32, device="cuda").view(1, -1),
        prefix_lengths=torch.tensor([5], dtype=torch.int32, device="cuda"),
        position_offsets=torch.tensor([5], dtype=torch.int32, device="cuda"),
    )
    provider = DensePrefixAttentionProvider(dense)
    query = torch.randn(7, 4, 64, dtype=torch.bfloat16, device="cuda")

    output, lse = provider.attend(layer_index=0, q=query, softmax_scale=64**-0.5)
    dense_key = key.repeat_interleave(2, dim=1).float()
    dense_value = value.repeat_interleave(2, dim=1).float()
    scores = torch.einsum("qhd,khd->qhk", query.float(), dense_key) * (64**-0.5)
    reference = torch.einsum("qhk,khd->qhd", scores.softmax(-1), dense_value)

    torch.testing.assert_close(output.float(), reference, atol=0.01, rtol=0.01)
    torch.testing.assert_close(lse.float(), scores.logsumexp(-1), atol=1e-4, rtol=1e-4)
