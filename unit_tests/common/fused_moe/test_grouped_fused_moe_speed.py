import torch
import time
import pytest
from lightllm.common.fused_moe.grouped_fused_moe import moe_align, moe_align1, grouped_matmul
from lightllm.utils.log_utils import init_logger

seed = 42
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

logger = init_logger(__name__)


@pytest.mark.parametrize("token_num", [200, 256, 8 * 1024])
def test_moe_align1(token_num):
    expert_num = 160
    topk_num = 6
    print(token_num)

    def get_one():
        rnd_logics = torch.randn(token_num, expert_num, device="cuda")
        topk_values, topk_ids = torch.topk(rnd_logics, topk_num, dim=1)

        experts_info = torch.zeros((expert_num, token_num * topk_num), dtype=torch.int32, device="cuda")
        experts_info.fill_(0)
        moe_align(topk_ids, experts_info)

        topk_weights = torch.randn((token_num, topk_num), dtype=torch.float32, device="cuda")
        experts_token_num = torch.zeros((expert_num,), dtype=torch.int32, device="cuda")
        experts_weights = torch.zeros(experts_info.shape, dtype=torch.float32, device="cuda")
        return experts_info, topk_weights, experts_weights, experts_token_num

    test_datas = [get_one() for _ in range(100)]

    moe_align1(*test_datas[0], topk_num)

    torch.cuda.synchronize()
    start = time.time()

    for i in range(60):
        moe_align1(*test_datas[i + 1], topk_num)
    torch.cuda.synchronize()

    print(f"token_num: {token_num} cost time: {time.time() - start} s")


if __name__ == "__main__":
    pytest.main()
