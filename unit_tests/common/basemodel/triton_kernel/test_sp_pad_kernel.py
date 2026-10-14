import torch
import time
import pytest
from lightllm.common.basemodel.triton_kernel.sp_pad_copy import sp_pad_copy
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@pytest.mark.parametrize(
    "token_num, hidden_dim, sp_world_size",
    [
        (token_num, hidden_dim, sp_world_size)
        for token_num in range(3, 6)
        for hidden_dim in [257, 2048]
        for sp_world_size in range(2, 5)
    ],
)
def test_sp_pad_copy(token_num, hidden_dim, sp_world_size):

    in_tensor = torch.randn((token_num, hidden_dim), dtype=torch.float16, device="cuda")
    out_tensors = [
        sp_pad_copy(in_tensor=in_tensor, sp_rank_id=rank_id, sp_world_size=sp_world_size)
        for rank_id in range(sp_world_size)
    ]
    out_tensor = torch.cat(out_tensors, dim=0)
    assert torch.equal(in_tensor, out_tensor[0:token_num, :])


if __name__ == "__main__":
    pytest.main()
