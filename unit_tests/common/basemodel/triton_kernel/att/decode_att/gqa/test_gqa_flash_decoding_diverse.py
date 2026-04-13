import pytest
import torch


def alloc_tensor_func(shape, dtype, device):
    return torch.empty(shape, dtype=dtype, device=device)


class MockReqManager:
    def __init__(self, req_to_token_indexs):
        self.req_to_token_indexs = req_to_token_indexs


class MockInferState:
    def __init__(
        self,
        batch_size,
        max_kv_seq_len,
        req_to_tokens,
        b_req_idx,
        b_seq_len,
        b_shared_seq_len=None,
        b_mark_shared_group=None,
    ):
        self.batch_size = batch_size
        self.max_kv_seq_len = max_kv_seq_len
        self.req_manager = MockReqManager(req_to_tokens)
        self.b_req_idx = b_req_idx
        self.b_seq_len = b_seq_len
        self.b_shared_seq_len = b_shared_seq_len
        self.b_mark_shared_group = b_mark_shared_group


@pytest.mark.parametrize("shared_seq_len", [0, 77, 256, 512])
@pytest.mark.parametrize("batch_size", [6, 18, 48, 96])
def test_gqa_flash_decoding_diverse_vs_baseline(shared_seq_len, batch_size):
    from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.flash_decoding.gqa_flash_decoding import (
        gqa_token_decode_attention_flash_decoding as baseline_attention,
    )
    from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.flash_decoding.gqa_flash_decoding_diverse import (
        gqa_token_decode_attention_flash_decoding_diverse as diverse_attention,
    )

    num_heads = 32
    kv_head_num = 8
    mark_shared_group_size = 3
    seq_len = 3547
    head_dim = 128
    max_len_in_batch = 4096
    test_dtype = torch.bfloat16

    kv_shape = (batch_size * max_len_in_batch, kv_head_num, head_dim)
    q = torch.randn(size=(batch_size, num_heads, head_dim), dtype=test_dtype, device="cuda")
    cache_k = torch.randn(size=kv_shape, dtype=test_dtype, device="cuda")
    cache_v = torch.randn(size=kv_shape, dtype=test_dtype, device="cuda")

    req_to_tokens = torch.arange(0, max_len_in_batch * batch_size, dtype=torch.int32, device="cuda").view(
        batch_size, max_len_in_batch
    )
    for i in range(batch_size):
        if i % mark_shared_group_size != 0:
            req_to_tokens[i, :shared_seq_len] = req_to_tokens[i - 1, :shared_seq_len]

    b_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
    b_shared_seq_len = torch.full((batch_size,), shared_seq_len, dtype=torch.int32, device="cuda")
    b_mark_shared_group = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
    b_mark_shared_group[mark_shared_group_size - 1 :: mark_shared_group_size] = mark_shared_group_size

    baseline_infer_state = MockInferState(
        batch_size=batch_size,
        max_kv_seq_len=max_len_in_batch,
        req_to_tokens=req_to_tokens,
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
    )
    diverse_infer_state = MockInferState(
        batch_size=batch_size,
        max_kv_seq_len=max_len_in_batch,
        req_to_tokens=req_to_tokens,
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_shared_seq_len=b_shared_seq_len,
        b_mark_shared_group=b_mark_shared_group,
    )

    baseline_out = baseline_attention(
        q=q.clone(),
        infer_state=baseline_infer_state,
        cache_k=cache_k,
        cache_v=cache_v,
        alloc_tensor_func=alloc_tensor_func,
    )
    diverse_out = diverse_attention(
        q=q.clone(),
        infer_state=diverse_infer_state,
        cache_k=cache_k,
        cache_v=cache_v,
        alloc_tensor_func=alloc_tensor_func,
    )

    assert torch.allclose(baseline_out, diverse_out, atol=1e-2, rtol=1e-2)

