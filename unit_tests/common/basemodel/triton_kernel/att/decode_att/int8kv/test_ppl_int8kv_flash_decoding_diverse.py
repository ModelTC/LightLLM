import pytest

pytest.skip(reason="need install lightllmKernel", allow_module_level=True)

import torch
from lightllm.utils.light_utils import light_ops


def alloc_tensor_func(shape, dtype, device):
    """兼容的 tensor 分配函数"""
    return torch.empty(shape, dtype=dtype, device=device)


class MockReqManager:
    """Mock request manager for testing"""

    def __init__(self, req_to_token_indexs):
        self.req_to_token_indexs = req_to_token_indexs


class MockInferState:
    """Mock infer state for testing"""

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


# @pytest.mark.parametrize("shared_seq_len", [512])
@pytest.mark.parametrize("shared_seq_len", [0, 77, 256, 311, 512, 550])
def test_token_decode_attention_flash_decoding_diverse_vs_baseline(shared_seq_len):
    """
    测试 ppl_int8kv_flash_decoding_diverse 的 token_decode_attention_flash_decoding
    与 ppl_int8kv_flash_decoding (baseline) 的对比。
    """

    from lightllm.common.basemodel.triton_kernel.att.decode_att.int8kv.ppl_int8kv_flash_decoding_diverse import (
        token_decode_attention_flash_decoding as diverse_attention,
    )
    from lightllm.common.basemodel.triton_kernel.att.decode_att.int8kv.ppl_int8kv_flash_decoding import (
        token_decode_attention_flash_decoding as baseline_attention,
    )

    batch_size = 6
    num_heads = 32
    kv_head_num = 8
    mark_shared_group_size = 3
    seq_len = 1024
    head_dim = 128
    quant_group_size = 8
    test_dtype = torch.bfloat16

    # 创建测试数据
    kv_shape = (batch_size * seq_len, kv_head_num, head_dim)
    kv_scale_shape = (batch_size * seq_len, kv_head_num, head_dim // quant_group_size)

    q = torch.randn(size=(batch_size, num_heads, head_dim), dtype=test_dtype, device="cuda")

    # 生成 cache_k 和 cache_v，使得每 mark_shared_group_size 个 batch 共享相同的 cache

    cache_k = torch.randint(low=-100, high=100, size=kv_shape, dtype=torch.int8, device="cuda")
    cache_k_scale = torch.ones(size=kv_scale_shape, dtype=test_dtype, device="cuda") / 100.0
    cache_v = torch.randint(low=-100, high=100, size=kv_shape, dtype=torch.int8, device="cuda")
    cache_v_scale = torch.ones(size=kv_scale_shape, dtype=test_dtype, device="cuda") / 100.0

    req_to_tokens = torch.arange(0, seq_len * batch_size, dtype=torch.int32, device="cuda").view(batch_size, seq_len)
    for i in range(batch_size):
        if i % mark_shared_group_size != 0:
            req_to_tokens[i, :shared_seq_len] = req_to_tokens[i - 1, :shared_seq_len]

    b_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
    b_shared_seq_len = torch.full((batch_size,), shared_seq_len, dtype=torch.int32, device="cuda")
    b_mark_shared_group = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
    b_mark_shared_group[mark_shared_group_size - 1 :: mark_shared_group_size] = mark_shared_group_size

    # 创建 baseline 的 infer_state (不需要 b_shared_seq_len)
    baseline_infer_state = MockInferState(
        batch_size=batch_size,
        max_kv_seq_len=seq_len,
        req_to_tokens=req_to_tokens,
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
    )

    # 创建 diverse 的 infer_state
    diverse_infer_state = MockInferState(
        batch_size=batch_size,
        max_kv_seq_len=seq_len,
        req_to_tokens=req_to_tokens,
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_shared_seq_len=b_shared_seq_len,
        b_mark_shared_group=b_mark_shared_group,
    )

    # 运行 baseline
    baseline_out = baseline_attention(
        q=q.clone(),
        infer_state=baseline_infer_state,
        cache_k=cache_k,
        cache_k_scale=cache_k_scale,
        cache_v=cache_v,
        cache_v_scale=cache_v_scale,
        alloc_tensor_func=alloc_tensor_func,
    )
    # 运行 diverse 版本
    diverse_out = diverse_attention(
        q=q.clone(),
        infer_state=diverse_infer_state,
        cache_k=cache_k,
        cache_k_scale=cache_k_scale,
        cache_v=cache_v,
        cache_v_scale=cache_v_scale,
        alloc_tensor_func=alloc_tensor_func,
    )

    print(f"\nshared_seq_len={shared_seq_len}")
    print(f"baseline_out: {baseline_out[0, 0, :4]}")
    print(f"diverse_out: {diverse_out[0, 0, :4]}")
    print(f"max diff: {(baseline_out - diverse_out).abs().max()}")

    # 与 baseline 对比
    assert torch.allclose(
        baseline_out, diverse_out, atol=1e-2, rtol=1e-2
    ), f"Diverse attention output should match baseline for shared_seq_len={shared_seq_len}"
