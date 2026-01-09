import torch
import time
import pytest
from lightllm.common.basemodel.triton_kernel.att.prefill_att.context_flashattention_nopad import (
    context_attention_fwd_contiguous_kv,
)
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def torch_context_attention_fwd2(q, k, v, o, b_start_loc, b_kv_start_loc, b_seq_len, b_prompt_cache_len):
    batch = b_start_loc.shape[0]

    for i in range(batch):
        start_loc = b_start_loc[i]
        kv_start_loc = b_kv_start_loc[i]
        seq_len = b_seq_len[i]
        prompt_cache_len = b_prompt_cache_len[i]
        cur_q = q[start_loc : start_loc + seq_len - prompt_cache_len, :, :]
        cur_q = cur_q.clone().to(torch.float32)
        cur_k = k[kv_start_loc : (kv_start_loc + seq_len), :, :]
        cur_k = cur_k.clone().to(torch.float32)

        cur_v = v[kv_start_loc : (kv_start_loc + seq_len), :, :]
        cur_v = cur_v.clone().to(torch.float32)

        dk = cur_q.shape[-1]
        cur_q = cur_q.permute(1, 0, 2)
        cur_k = cur_k.permute(1, 2, 0)
        cur_v = cur_v.permute(1, 0, 2)
        dk = cur_q.shape[-1]

        p = torch.matmul(cur_q, cur_k) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

        q_index = (torch.arange(cur_q.shape[1]).to(p.device) + prompt_cache_len).view(-1, 1)
        k_index = torch.arange(seq_len).to(p.device).view(1, -1)

        p[:, (q_index < k_index)] = float("-inf")

        s = torch.nn.functional.softmax(p, dim=-1)

        o[start_loc : start_loc + seq_len - prompt_cache_len, :, :] = torch.matmul(s, cur_v).transpose(0, 1)


@pytest.mark.parametrize(
    "B, H, N_CTX, D_HEAD, prompt_cache_len",
    [
        (b, H, N_CTX, D_HEAD, prompt_cache_len)
        for b in [1, 2, 4]
        for H in [1, 8]
        for N_CTX in [3, 10, 1024]
        for D_HEAD in [64, 128]
        for prompt_cache_len in [0, 56, 200]
    ],
)
def test_context_attention_fwd_contiguous_kv(B, H, N_CTX, D_HEAD, prompt_cache_len):
    dtype = torch.float16
    prompt_cache_len = 0
    if prompt_cache_len >= N_CTX - 1:
        return

    q = torch.empty((B * (N_CTX - prompt_cache_len), H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    kv = torch.empty((B * N_CTX, 2 * H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2)
    k = kv[:, :H, :]
    v = kv[:, H:, :]

    o = torch.empty((B * (N_CTX - prompt_cache_len), H, D_HEAD), dtype=dtype, device="cuda")
    torch_o = torch.empty((B * (N_CTX - prompt_cache_len), H, D_HEAD), dtype=dtype, device="cuda")

    max_q_input_len = N_CTX - prompt_cache_len

    b_seq_len = torch.ones((B,), dtype=torch.int32, device="cuda")
    b_start_loc = torch.zeros((B,), dtype=torch.int32, device="cuda")
    b_seq_len = torch.ones((B,), dtype=torch.int32, device="cuda")
    b_prompt_cache_len = torch.zeros(B, dtype=torch.int32, device="cuda")

    for i in range(B):
        b_seq_len[i] = N_CTX
        if i != 0:
            b_start_loc[i] = b_start_loc[i - 1] + N_CTX - prompt_cache_len
        b_prompt_cache_len[i] = prompt_cache_len

    b_kv_start_loc = torch.cumsum(b_seq_len, dim=0, dtype=torch.int32) - b_seq_len
    torch_context_attention_fwd2(q, k, v, torch_o, b_start_loc, b_kv_start_loc, b_seq_len, b_prompt_cache_len)
    context_attention_fwd_contiguous_kv(
        q=q,
        k=k,
        v=v,
        o=o,
        b_start_loc=b_start_loc,
        b_kv_start_loc=b_kv_start_loc,
        b_seq_len=b_seq_len,
        max_q_input_len=max_q_input_len,
        b_prompt_cache_len=b_prompt_cache_len,
    )

    assert torch.allclose(torch_o, o, atol=1e-2, rtol=0)
    cos = torch.nn.CosineSimilarity(0)
    assert cos(o.flatten().float(), torch_o.flatten().float()) > 0.99


if __name__ == "__main__":
    pytest.main()
