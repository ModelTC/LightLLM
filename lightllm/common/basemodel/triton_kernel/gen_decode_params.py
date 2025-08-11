import torch
import triton
import triton.language as tl
from .gen_prefill_params import gen_cumsum_pad0_tensor
from lightllm.utils.envs_utils import get_env_start_args


@torch.no_grad()
def gen_decode_params(b_seq_len: torch.Tensor):
    b_kv_seq_len = b_seq_len
    position_ids = b_seq_len - 1
    mtp_step = get_env_start_args().mtp_step
    mtp_size = mtp_step + 1
    enable_fa3_mtp = get_env_start_args().enable_fa3_mtp

    if enable_fa3_mtp:
        b_q_seq_len = torch.ones_like(b_seq_len[: len(b_seq_len) // mtp_size])
        b1_cu_q_seq_len, b1_cu_kv_seq_len = gen_cumsum_pad0_tensor(b_q_seq_len, b_kv_seq_len[mtp_size - 1 :: mtp_size])
    else:
        b_q_seq_len = torch.ones_like(b_seq_len)
        b1_cu_q_seq_len, b1_cu_kv_seq_len = gen_cumsum_pad0_tensor(b_q_seq_len, b_kv_seq_len)

    return b_q_seq_len, b1_cu_q_seq_len, b_kv_seq_len, b1_cu_kv_seq_len, position_ids
