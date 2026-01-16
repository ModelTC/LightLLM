import pytest

pytest.skip(reason="need install lightllmkernel", allow_module_level=True)

import torch
from lightllm.utils.light_utils import light_ops


def create_tensors(shared_seq_len):
    batch_size = 4
    num_heads = 32
    kv_head_num = 8
    seq_len = 256
    head_dim = 128
    max_len_in_batch = seq_len
    block_seq = 256
    max_batch_group_size = 4
    quant_group_size = 8

    test_dtype = torch.bfloat16

    kv_shape = (batch_size * seq_len, kv_head_num, head_dim)
    kv_scale_shape = (batch_size * seq_len, kv_head_num, head_dim // quant_group_size)

    q = torch.randn(size=(batch_size, num_heads, head_dim), dtype=test_dtype, device="cuda")
    k = torch.randint(low=-100, high=100, size=kv_shape, dtype=torch.int8, device="cuda")
    k_scale = torch.ones(size=kv_scale_shape, dtype=test_dtype, device="cuda")
    v = torch.randint(low=-100, high=100, size=kv_shape, dtype=torch.int8, device="cuda")
    v_scale = torch.ones(size=kv_scale_shape, dtype=test_dtype, device="cuda")
    Req_to_tokens = torch.arange(0, seq_len * batch_size, dtype=torch.int32, device="cuda").view(batch_size, seq_len)
    B_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
    b_shared_seq_len = torch.full((batch_size,), shared_seq_len, dtype=torch.int32, device="cuda")
    b_mark_shared_group = torch.ones(batch_size, dtype=torch.int32, device="cuda")
    mid_out = torch.zeros(
        size=(batch_size, num_heads, (seq_len // block_seq) + 2, head_dim), dtype=q.dtype, device="cuda"
    )
    mid_out_logsumexp = torch.zeros(
        size=(batch_size, num_heads, (seq_len // block_seq) + 2), dtype=q.dtype, device="cuda"
    )

    return {
        "q": q,
        "k": k,
        "k_scale": k_scale,
        "v": v,
        "v_scale": v_scale,
        "Req_to_tokens": Req_to_tokens,
        "B_req_idx": B_req_idx,
        "b_seq_len": b_seq_len,
        "b_shared_seq_len": b_shared_seq_len,
        "b_mark_shared_group": b_mark_shared_group,
        "max_len_in_batch": max_len_in_batch,
        "mid_out": mid_out,
        "mid_out_logsumexp": mid_out_logsumexp,
        "block_seq": block_seq,
        "max_batch_group_size": max_batch_group_size,
        "head_dim": head_dim,
    }


@pytest.mark.parametrize("shared_seq_len", [0, 47, 77, 128, 200, 255])
def test_flash_decode_stage2_execution(shared_seq_len):
    setup_tensors = create_tensors(shared_seq_len)

    light_ops.group8_int8kv_flashdecoding_diverse_stage2(
        setup_tensors["block_seq"],
        setup_tensors["mid_out"],
        setup_tensors["mid_out_logsumexp"],
        1.0 / (setup_tensors["head_dim"] ** 0.5),
        setup_tensors["q"],
        setup_tensors["k"],
        setup_tensors["k_scale"],
        setup_tensors["v"],
        setup_tensors["v_scale"],
        setup_tensors["Req_to_tokens"],
        setup_tensors["B_req_idx"],
        setup_tensors["b_seq_len"],
        setup_tensors["b_shared_seq_len"],
        setup_tensors["max_len_in_batch"],
    )
    seq_block_idx = (setup_tensors["b_shared_seq_len"][0].item() + setup_tensors["block_seq"] - 1) // setup_tensors[
        "block_seq"
    ]
    mid_out = setup_tensors["mid_out"][:, :, seq_block_idx:, :]
    mid_out_logsumexp = setup_tensors["mid_out_logsumexp"][:, :, seq_block_idx:]

    q = setup_tensors["q"]
    k = setup_tensors["k"]
    v = setup_tensors["v"]
    true_mid_out = torch.zeros_like(mid_out)
    true_mid_out_logsumexp = torch.zeros_like(mid_out_logsumexp)
    new_q = q
    new_k = k.to(q.dtype)
    new_v = v.to(q.dtype)

    b_seq_len = setup_tensors["b_seq_len"] - setup_tensors["b_shared_seq_len"]
    req_to_tokens = setup_tensors["Req_to_tokens"][:, setup_tensors["b_shared_seq_len"][0].item() :]

    from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.flash_decoding.gqa_flash_decoding_stage1 import (
        flash_decode_stage1 as gqa_flash_decode_stage1,
    )

    gqa_flash_decode_stage1(
        q=new_q,
        k=new_k,
        v=new_v,
        Req_to_tokens=req_to_tokens,
        B_req_idx=setup_tensors["B_req_idx"],
        B_Seqlen=b_seq_len,
        max_len_in_batch=setup_tensors["max_len_in_batch"],
        mid_out=true_mid_out,
        mid_out_logsumexp=true_mid_out_logsumexp,
        block_seq=setup_tensors["block_seq"],
    )
    print(f"\nshared_seq_len={shared_seq_len}")
    print(f"mid_out: {mid_out[0:4, 0, 0, 0]}")
    print(f"true_mid_out: {true_mid_out[0:4, 0, 0, 0]}")
    abs_diff = (mid_out - true_mid_out).abs()
    max_diff = abs_diff.max()
    max_diff_idx = abs_diff.argmax()
    max_diff_idx_unraveled = torch.unravel_index(max_diff_idx, abs_diff.shape)
    mid_out_value = mid_out[max_diff_idx_unraveled]
    true_mid_out_value = true_mid_out[max_diff_idx_unraveled]
    print(f"max abs diff: {max_diff}, mid_out value: {mid_out_value}, " f"true_mid_out value: {true_mid_out_value}")

    assert torch.allclose(
        mid_out[0:4, 0, 0, 0], true_mid_out[0:4, 0, 0, 0], atol=1e-2
    ), f"Mid output does not match expected values for shared_seq_len={shared_seq_len}"
    assert torch.allclose(
        mid_out_logsumexp, true_mid_out_logsumexp, atol=1e-2
    ), f"LogSumExp output does not match expected values for shared_seq_len={shared_seq_len}"
