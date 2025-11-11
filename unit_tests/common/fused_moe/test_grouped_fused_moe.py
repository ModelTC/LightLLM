import torch
import time
import pytest
import triton
from lightllm.common.fused_moe.grouped_fused_moe import moe_align, moe_align1, moe_align2, grouped_matmul
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

seed = 42
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test_moe_align():
    expert_num = 5
    token_num = 3
    topk = 3
    topk_ids = torch.tensor([[0, 1, 2], [0, 3, 1], [3, 1, 4]], dtype=torch.int32, device="cuda")
    out = torch.zeros((expert_num, token_num * topk), dtype=torch.int32, device="cuda")
    out.fill_(0)
    moe_align(topk_ids, out)
    true = torch.tensor(
        [
            [1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    assert torch.equal(out, true)


def test_moe_align1():
    experts_info = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
        dtype=torch.int32,
        device="cuda",
    )
    topk_weights = torch.tensor([[0.3, 0.7], [0.2, 0.8]], dtype=torch.float32, device="cuda")
    experts_token_num = torch.zeros((4,), dtype=torch.int32, device="cuda")
    experts_weights = torch.zeros(experts_info.shape, dtype=torch.float32, device="cuda")

    moe_align1(experts_info, topk_weights, experts_weights, experts_token_num, 2)

    true_experts_token_num = torch.tensor([1, 2, 1, 0], device="cuda", dtype=torch.int32)
    true_experts_info = torch.tensor(
        [[0, 0, 0, 0], [1, 2, 1, 0], [3, 0, 0, 1], [0, 0, 0, 0]], device="cuda:0", dtype=torch.int32
    )
    true_experts_weights = torch.tensor(
        [
            [0.3000, 0.0000, 0.0000, 0.0000],
            [0.7000, 0.2000, 0.0000, 0.0000],
            [0.8000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
        ],
        device="cuda",
        dtype=torch.float32,
    )

    assert torch.allclose(true_experts_weights, experts_weights)
    assert torch.equal(experts_token_num, true_experts_token_num)
    assert torch.equal(experts_info, true_experts_info)


def test_moe_align2():

    experts_token_num = torch.zeros((4,), dtype=torch.int32, device="cuda")
    experts_token_num[0] = 8
    experts_token_num[1] = 0
    experts_token_num[2] = 60
    experts_token_num[3] = 16

    mblocks_to_tuple_info = moe_align2(100, experts_token_num, block_m=16)
    assert mblocks_to_tuple_info.shape[0] == triton.cdiv(100 + 4 * (16 - 1), 16)
    assert torch.allclose(
        mblocks_to_tuple_info[:, 0],
        torch.tensor([0, 2, 2, 2, 2, 3, -1, -1, -1, -1], device="cuda", dtype=torch.int32),
    )
    assert torch.allclose(
        mblocks_to_tuple_info[:, 1], torch.tensor([0, 0, 1, 2, 3, 0, 0, 0, 0, 0], device="cuda", dtype=torch.int32)
    )


def test_grouped_matmul():
    test_dtype = torch.bfloat16
    token_inputs = torch.randn((10, 512), dtype=test_dtype, device="cuda") / 10
    experts_token_num = torch.tensor([1, 9], dtype=torch.int32, device="cuda")
    experts_to_token_index = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    experts_to_weights = torch.tensor(
        [
            [0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    expert_weights = torch.randn((2, 1024, 512), dtype=test_dtype, device="cuda") / 10
    topk_num = 1
    out = torch.empty((10, 1024), dtype=test_dtype, device="cuda")
    # warm up
    grouped_matmul(
        10 * 1,
        token_inputs,
        None,
        experts_token_num,
        experts_to_token_index,
        experts_to_weights,
        expert_weights,
        None,
        topk_num,
        out,
        mul_routed_weight=True,
        use_fp8_w8a8=False,
    )
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        grouped_matmul(
            10 * 1,
            token_inputs,
            None,
            experts_token_num,
            experts_to_token_index,
            experts_to_weights,
            expert_weights,
            None,
            topk_num,
            out,
            mul_routed_weight=True,
            use_fp8_w8a8=False,
        )
    torch.cuda.synchronize()
    logger.info(f"grouped_matmul test cost time: {time.time() - start} s")

    ans_list = []
    ans_list.append(torch.matmul(token_inputs[0:1, :], expert_weights[0].transpose(0, 1)))
    for i in range(9):
        t_ans = torch.matmul(token_inputs[(i + 1) : (i + 2), :], expert_weights[1].transpose(0, 1))
        ans_list.append(t_ans)

    true_out = torch.cat(ans_list, dim=0)

    assert torch.allclose(0.5 * true_out, out, atol=1e-2, rtol=0)


if __name__ == "__main__":
    pytest.main()
