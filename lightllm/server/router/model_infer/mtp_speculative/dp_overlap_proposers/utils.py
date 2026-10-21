import torch

from lightllm.common.basemodel.triton_kernel.mtp_utils import gen_b_req_mtp_start_loc


def get_dp_overlap_req_start_rows(b_mtp_index: torch.Tensor, req_num: int) -> torch.Tensor:
    """根据动态 verify 布局恢复请求起始行。"""

    req_num = int(req_num)
    if req_num == 0:
        assert b_mtp_index.numel() == 0
        return torch.empty((0,), dtype=torch.int32, device=b_mtp_index.device)
    assert b_mtp_index.is_cuda, "b_mtp_index must be a CUDA tensor"
    return gen_b_req_mtp_start_loc(
        b_mtp_index=b_mtp_index,
        num_reqs=req_num,
    )
