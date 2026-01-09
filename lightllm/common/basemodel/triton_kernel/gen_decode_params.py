import torch
import triton
import triton.language as tl
from .gen_prefill_params import gen_cumsum_pad0_tensor
from lightllm.utils.envs_utils import get_env_start_args


@torch.no_grad()
def gen_decode_params(b_seq_len: torch.Tensor):
    b_kv_seq_len = b_seq_len
    position_ids = b_seq_len - 1
    b_q_seq_len = torch.ones(b_seq_len.shape[0], dtype=torch.int32, device=b_seq_len.device)
    b1_cu_q_seq_len, b1_cu_kv_seq_len = gen_cumsum_pad0_tensor(b_q_seq_len, b_kv_seq_len)
    return b_q_seq_len, b1_cu_q_seq_len, b_kv_seq_len, b1_cu_kv_seq_len, position_ids
