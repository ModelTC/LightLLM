import torch
import time
import pytest
from lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int8kv_copy_kv import (
    dequantize_int8kv,
    destindex_copy_quantize_kv,
)
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def torch_dequant(kv, kv_scale, b_req_idx, b_seq_len, req_to_token_indexs, odtype, group_quant_size):
    batch = b_req_idx.shape[0]
    tmp_out = []
    for i in range(batch):
        req_idx = b_req_idx[i]
        seq_len = b_seq_len[i]
        kv_loc = req_to_token_indexs[req_idx, :seq_len]
        head_num = kv.shape[1]
        cur_kv = kv[kv_loc, :, :].reshape(seq_len, head_num, -1, group_quant_size).to(odtype)
        cur_scale = kv_scale[kv_loc, :, :].reshape(seq_len, head_num, -1, 1)
        out = cur_kv * cur_scale
        tmp_out.append(out.reshape(seq_len, head_num, -1))
    return torch.cat(tmp_out, dim=0)


@pytest.mark.parametrize(
    "B, H, N_CTX, D_HEAD, group_quant_size",
    [
        (b, H, N_CTX, D_HEAD, group_quant_size)
        for b in [1, 2, 4]
        for H in [1, 8]
        for N_CTX in [3, 10, 1024]
        for D_HEAD in [64, 128]
        for group_quant_size in [8, 16]
    ],
)
def test_dequantize_int8kv(B, H, N_CTX, D_HEAD, group_quant_size):
    dtype = torch.bfloat16
    kv = torch.empty((B * N_CTX, 2 * H, D_HEAD), dtype=torch.int8, device="cuda").random_(-10, 10)
    kv_scale = torch.randn((B * N_CTX, 2 * H, D_HEAD // group_quant_size), dtype=dtype, device="cuda")
    out = torch.empty((B * N_CTX, 2 * H, D_HEAD), dtype=dtype, device="cuda")
    req_to_token_indexs = torch.empty((B, N_CTX), dtype=torch.int32, device="cuda")
    max_input_len = N_CTX
    b_seq_len = torch.ones((B,), dtype=torch.int32, device="cuda")
    b_seq_len.fill_(N_CTX)
    b_req_idx = torch.arange(0, B, dtype=torch.int32, device="cuda")
    req_to_token_indexs.view(-1)[:] = torch.arange(0, B * N_CTX, dtype=torch.int32, device="cuda")
    b_kv_start_loc = torch.cumsum(b_seq_len, dim=0, dtype=torch.int32) - b_seq_len

    k = kv[:, :H, :]
    v = kv[:, H:, :]
    k_scale = kv_scale[:, :H, :]
    v_scale = kv_scale[:, H:, :]

    ground_out = torch_dequant(
        kv=kv,
        kv_scale=kv_scale,
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        req_to_token_indexs=req_to_token_indexs,
        odtype=out.dtype,
        group_quant_size=group_quant_size,
    )
    dequantize_int8kv(
        k=k,
        k_scale=k_scale,
        v=v,
        v_scale=v_scale,
        req_to_token_indexs=req_to_token_indexs,
        b_seq_len=b_seq_len,
        b_req_idx=b_req_idx,
        b_kv_start_loc=b_kv_start_loc,
        k_out=out[:, :H, :],
        v_out=out[:, H:, :],
        max_len_in_batch=max_input_len,
        quant_group_size=group_quant_size,
    )
    assert torch.allclose(out, ground_out, atol=1e-2, rtol=0)
    cos = torch.nn.CosineSimilarity(0)
    assert cos(out.flatten().float(), ground_out.flatten().float()) > 0.99


if __name__ == "__main__":
    pytest.main()
