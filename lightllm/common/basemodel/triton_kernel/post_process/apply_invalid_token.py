import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_apply_invalid_token(
    Logits,
    invalid_token_ids,
    cu_invalid_token_num,
    stride_logit_b,
):
    cur_batch = tl.program_id(0)
    start_index = tl.load(cu_invalid_token_num + cur_batch)
    end_index = tl.load(cu_invalid_token_num + cur_batch + 1)
    for i in range(start_index, end_index):
        cur_invalid_token_id = tl.load(invalid_token_ids + i)
        cur_logit_ptr = Logits + cur_batch * stride_logit_b + cur_invalid_token_id
        tl.store(cur_logit_ptr, float("-inf"))
    return


def apply_invalid_token_ids(
    Logits: torch.Tensor,
    invalid_token_ids: torch.Tensor,
    cu_invalid_token_num: torch.Tensor,
):
    batch_size = Logits.shape[0]
    grid = (batch_size,)
    _fwd_kernel_apply_invalid_token[grid](
        Logits=Logits,
        invalid_token_ids=invalid_token_ids,
        cu_invalid_token_num=cu_invalid_token_num,
        stride_logit_b=Logits.stride(0),
    )
    return
