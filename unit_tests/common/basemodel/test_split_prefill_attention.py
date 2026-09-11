from types import SimpleNamespace

import pytest
import torch

import lightllm.common.basemodel.attention.split_prefill.backend as backend_module
from lightllm.common.basemodel.attention.split_prefill import (
    DenseSplitPrefillAttBackend,
)
from lightllm.server.activation_transfer import KVActivation
from lightllm.server.generationserver.kv_materializer import (
    gather_dense_prefix_kv,
)


def test_split_prefill_backend_rejects_unsupported_dtype():
    model = SimpleNamespace(data_type=torch.float64)
    with pytest.raises(TypeError, match="does not support"):
        DenseSplitPrefillAttBackend(model)


def test_split_prefill_backend_is_owned_by_each_model():
    first = DenseSplitPrefillAttBackend(SimpleNamespace(data_type=torch.float32))
    second = DenseSplitPrefillAttBackend(SimpleNamespace(data_type=torch.float32))

    assert first is not second
    assert first.model is not second.model


def test_half_split_prefill_requires_fa3(monkeypatch):
    monkeypatch.setattr(backend_module, "flash_attn_with_kvcache", None)
    model = SimpleNamespace(data_type=torch.bfloat16)

    with pytest.raises(RuntimeError, match="flash_attn_with_kvcache"):
        DenseSplitPrefillAttBackend(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("first_suffix_token_isolated", [True, False])
def test_fp32_split_prefill_matches_dense_without_host_sync(
    first_suffix_token_isolated,
):
    torch.manual_seed(17)
    device = torch.device("cuda")
    batch_size, suffix_length = 2, 4
    query_heads, head_dim = 4, 32
    req_to_token = torch.tensor(
        [
            [7, 2, 11, 0, 0, 0, 0, 0],
            [5, 13, 3, 9, 1, 0, 0, 0],
        ],
        dtype=torch.int32,
        device=device,
    )
    prefix_key = torch.randn(16, 1, head_dim, device=device)
    prefix_value = torch.randn_like(prefix_key)
    memory = SimpleNamespace(
        layer_num=1,
        get_att_input_params=lambda _layer: (prefix_key, prefix_value),
    )
    dense_prefix = gather_dense_prefix_kv(
        memory,
        req_to_token,
        (
            KVActivation(1, 11, 0, 3),
            KVActivation(2, 12, 1, 5),
        ),
    )
    model = SimpleNamespace(
        data_type=torch.float32,
        trans_layers_weight=[SimpleNamespace(local_q_heads=query_heads)],
    )
    infer_state = SimpleNamespace(
        batch_size=batch_size,
        suffix_length=suffix_length,
        prefix_kv=dense_prefix,
        prefix_lengths=dense_prefix.prefix_lengths,
        first_suffix_token_isolated=first_suffix_token_isolated,
    )
    query = torch.randn(
        batch_size * suffix_length,
        query_heads,
        head_dim,
        device=device,
    )
    suffix_key = torch.randn(batch_size * suffix_length, 1, head_dim, device=device)
    suffix_value = torch.randn_like(suffix_key)

    backend = DenseSplitPrefillAttBackend(model)
    state = backend.create_att_prefill_state(infer_state)
    previous_sync_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        state.init_state()
        page_table_ptr = state.page_table.data_ptr()
        cu_seqlens_ptr = state.cu_seqlens_q.data_ptr()
        state.init_state()
        assert state.page_table.data_ptr() == page_table_ptr
        assert state.cu_seqlens_q.data_ptr() == cu_seqlens_ptr
        output = state.prefill_att(
            query,
            dense_prefix.layer(0).key,
            dense_prefix.layer(0).value,
            suffix_k=suffix_key,
            suffix_v=suffix_value,
        )
    finally:
        torch.cuda.set_sync_debug_mode(previous_sync_mode)

    query = query.view(batch_size, suffix_length, query_heads, head_dim)
    suffix_key = suffix_key.view(batch_size, suffix_length, 1, head_dim)
    suffix_value = suffix_value.view(batch_size, suffix_length, 1, head_dim)
    references = []
    for batch_index, prefix_length in enumerate((3, 5)):
        indexes = req_to_token[batch_index, :prefix_length].long()
        key = torch.cat((prefix_key[indexes], suffix_key[batch_index]), dim=0).expand(
            -1, query_heads, -1
        )
        value = torch.cat(
            (prefix_value[indexes], suffix_value[batch_index]), dim=0
        ).expand(-1, query_heads, -1)
        scores = torch.einsum("qhd,khd->qhk", query[batch_index], key) * (
            head_dim**-0.5
        )
        if first_suffix_token_isolated:
            scores[0, :, prefix_length + 1 :] = -torch.inf
        references.append(torch.einsum("qhk,khd->qhd", scores.softmax(-1), value))
    reference = torch.stack(references).reshape_as(output)
    torch.testing.assert_close(output, reference, atol=2e-5, rtol=2e-5)
