import pytest
import torch

from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.flash_decoding.gqa_flash_decoding_diverse_stage2 import (
    flash_decode_stage2,
)


def create_tensors(
    shared_seq_len,
    batch_size=6,
    seq_len=1024,
    max_len_in_batch=4096,
    max_batch_group_size=3,
):
    num_heads = 32
    kv_head_num = 8
    head_dim = 128
    block_seq = 256
    test_dtype = torch.bfloat16

    kv_shape = (batch_size * max_len_in_batch, kv_head_num, head_dim)
    q = torch.randn(size=(batch_size, num_heads, head_dim), dtype=test_dtype, device="cuda")
    k = torch.randn(size=kv_shape, dtype=test_dtype, device="cuda")
    v = torch.randn(size=kv_shape, dtype=test_dtype, device="cuda")
    req_to_tokens = torch.arange(0, max_len_in_batch * batch_size, dtype=torch.int32, device="cuda").view(
        batch_size, max_len_in_batch
    )
    b_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
    b_shared_seq_len = torch.full((batch_size,), shared_seq_len, dtype=torch.int32, device="cuda")
    b_mark_shared_group = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
    b_mark_shared_group[max_batch_group_size - 1 :: max_batch_group_size] = max_batch_group_size
    mid_out = torch.zeros(
        size=(batch_size, num_heads, (max_len_in_batch // block_seq) + 2, head_dim), dtype=q.dtype, device="cuda"
    )
    mid_out_logsumexp = torch.zeros(
        size=(batch_size, num_heads, (max_len_in_batch // block_seq) + 2), dtype=torch.float32, device="cuda"
    )

    for i in range(batch_size):
        if i % max_batch_group_size != 0:
            req_to_tokens[i, :shared_seq_len] = req_to_tokens[i - 1, :shared_seq_len]

    return {
        "q": q,
        "k": k,
        "v": v,
        "Req_to_tokens": req_to_tokens,
        "B_req_idx": b_req_idx,
        "b_seq_len": b_seq_len,
        "b_shared_seq_len": b_shared_seq_len,
        "b_mark_shared_group": b_mark_shared_group,
        "max_len_in_batch": max_len_in_batch,
        "mid_out": mid_out,
        "mid_out_logsumexp": mid_out_logsumexp,
        "block_seq": block_seq,
    }


@pytest.mark.parametrize("shared_seq_len", [0, 47, 77, 128, 255])
def test_flash_decode_stage2_execution(shared_seq_len):
    setup_tensors = create_tensors(shared_seq_len)

    flash_decode_stage2(
        q=setup_tensors["q"],
        k=setup_tensors["k"],
        v=setup_tensors["v"],
        Req_to_tokens=setup_tensors["Req_to_tokens"],
        B_req_idx=setup_tensors["B_req_idx"],
        B_Seqlen=setup_tensors["b_seq_len"],
        b_shared_seq_len=setup_tensors["b_shared_seq_len"],
        max_len_in_batch=setup_tensors["max_len_in_batch"],
        mid_out=setup_tensors["mid_out"],
        mid_out_logsumexp=setup_tensors["mid_out_logsumexp"],
        block_seq=setup_tensors["block_seq"],
    )

    seq_block_idx = (setup_tensors["b_shared_seq_len"][0].item() + setup_tensors["block_seq"] - 1) // setup_tensors[
        "block_seq"
    ]
    mid_out = setup_tensors["mid_out"][:, :, seq_block_idx:, :]
    mid_out_logsumexp = setup_tensors["mid_out_logsumexp"][:, :, seq_block_idx:]

    true_mid_out = torch.zeros_like(mid_out)
    true_mid_out_logsumexp = torch.zeros_like(mid_out_logsumexp)
    b_seq_len = setup_tensors["b_seq_len"] - setup_tensors["b_shared_seq_len"]
    req_to_tokens = setup_tensors["Req_to_tokens"][:, setup_tensors["b_shared_seq_len"][0].item() :]

    from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.flash_decoding.gqa_flash_decoding_stage1 import (
        flash_decode_stage1 as gqa_flash_decode_stage1,
    )

    gqa_flash_decode_stage1(
        q=setup_tensors["q"],
        k=setup_tensors["k"],
        v=setup_tensors["v"],
        Req_to_tokens=req_to_tokens,
        B_req_idx=setup_tensors["B_req_idx"],
        B_Seqlen=b_seq_len,
        max_len_in_batch=setup_tensors["max_len_in_batch"],
        mid_out=true_mid_out,
        mid_out_logsumexp=true_mid_out_logsumexp,
        block_seq=setup_tensors["block_seq"],
    )

    assert torch.allclose(mid_out, true_mid_out, atol=1e-2, rtol=1e-2)
    assert torch.allclose(mid_out_logsumexp, true_mid_out_logsumexp, atol=1e-2, rtol=1e-2)

