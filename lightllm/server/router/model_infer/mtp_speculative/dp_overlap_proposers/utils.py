import torch

from lightllm.backends import get_device_runtime, get_operator_backend

_device_runtime = get_device_runtime()
_operator_backend = get_operator_backend()


gen_b_req_mtp_start_loc = _operator_backend.resolve("common.basemodel.mtp_utils.gen_b_req_mtp_start_loc")


def get_dp_overlap_req_start_rows(b_mtp_index: torch.Tensor, req_num: int) -> torch.Tensor:
    """根据动态 verify 布局恢复请求起始行。"""

    req_num = int(req_num)
    if req_num == 0:
        assert b_mtp_index.numel() == 0
        return _device_runtime.empty((0,), dtype=torch.int32, device=b_mtp_index.device)
    assert _device_runtime.is_device_tensor(b_mtp_index), "b_mtp_index must be a CUDA tensor"
    return gen_b_req_mtp_start_loc(
        b_mtp_index=b_mtp_index,
        num_reqs=req_num,
    )
