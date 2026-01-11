import torch
import pytest
import numpy as np
from typing import Tuple
from lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int4kv_copy_kv import destindex_copy_int4kv, dequantize_int4kv
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def test_quanted_and_dequant():
    """Test quantization followed by dequantization."""
    batch_size = 1
    seq_len = 8
    head_num = 4
    k_head_num = 2
    v_head_num = 2
    assert k_head_num + v_head_num == head_num
    head_dim = 64
    quant_group_size = 8

    # Create original data
    original_kv = torch.randn(batch_size * seq_len, head_num, head_dim, dtype=torch.float32).clamp_(-1, 1).cuda()
    dest_loc = torch.arange(batch_size * seq_len, dtype=torch.int64).cuda()

    # Quantize
    group_count = head_dim // quant_group_size
    kv_buffer = torch.zeros(batch_size * seq_len, head_num, head_dim // 2, dtype=torch.int8).cuda()
    kv_scale_buffer = torch.zeros(batch_size * seq_len, head_num, group_count, dtype=torch.float32).cuda()
    destindex_copy_int4kv(original_kv, dest_loc, kv_buffer, kv_scale_buffer, quant_group_size)

    # Dequantize
    req_to_token_indexs = torch.arange(seq_len, dtype=torch.int64).view(1, -1).cuda()
    b_seq_len = torch.tensor([seq_len], dtype=torch.int32).cuda()
    b_req_idx = torch.tensor([0], dtype=torch.int32).cuda()
    b_kv_start_loc = torch.tensor([0], dtype=torch.int32).cuda()

    recovered_kv = torch.zeros(batch_size * seq_len, head_num, head_dim, dtype=torch.float32).cuda()

    dequantize_int4kv(
        k=kv_buffer[:, 0:k_head_num, :],
        k_scale=kv_scale_buffer[:, 0:k_head_num, :],
        v=kv_buffer[:, k_head_num:, :],
        v_scale=kv_scale_buffer[:, k_head_num:, :],
        req_to_token_indexs=req_to_token_indexs,
        b_seq_len=b_seq_len,
        b_req_idx=b_req_idx,
        b_kv_start_loc=b_kv_start_loc,
        k_out=recovered_kv[:, :k_head_num, :],
        v_out=recovered_kv[:, k_head_num:, :],
        max_len_in_batch=seq_len,
        quant_group_size=quant_group_size,
    )

    logger.info("Round-trip test completed!")
    assert torch.allclose(recovered_kv, original_kv, atol=2 / 14, rtol=0)
    cos = torch.nn.CosineSimilarity(0)
    assert cos(recovered_kv.flatten().float(), original_kv.flatten().float()) > 0.99


if __name__ == "__main__":
    pytest.main()
