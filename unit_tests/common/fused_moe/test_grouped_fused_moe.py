import torch
import time
import pytest
import triton
from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe import (
    _moe_align_fused_atomic_token,
    fused_experts_impl,
    moe_align,
    moe_align_fused,
    moe_align1,
    moe_align2,
    grouped_matmul,
)
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


def _check_moe_align_fused(topk_ids, topk_weights, expert_num, ordered=True):
    expert_to_token_index = torch.empty((expert_num, topk_ids.numel()), dtype=torch.int32, device="cuda")
    expert_to_weight = torch.empty((expert_num, topk_ids.numel()), dtype=torch.float32, device="cuda")
    expert_token_num = torch.empty((expert_num,), dtype=torch.int32, device="cuda")

    moe_align_fused(
        expert_to_token_index,
        expert_to_weight,
        expert_token_num,
        topk_ids,
        topk_weights,
    )
    torch.cuda.synchronize()

    flat_topk_ids = topk_ids.flatten()
    flat_topk_weights = topk_weights.flatten()
    expected_token_num = torch.bincount(flat_topk_ids, minlength=expert_num).to(torch.int32)
    assert torch.equal(expert_token_num, expected_token_num)

    for expert_id, token_num in enumerate(expected_token_num.tolist()):
        expected_index = torch.nonzero(flat_topk_ids == expert_id, as_tuple=False).flatten()
        expected_weight = flat_topk_weights[expected_index]
        expected_index = expected_index.to(torch.int32)
        token_index = expert_to_token_index[expert_id, :token_num]
        token_weight = expert_to_weight[expert_id, :token_num]

        if not ordered:
            order = torch.argsort(token_index)
            token_index = token_index[order]
            token_weight = token_weight[order]

        assert torch.equal(token_index, expected_index)
        assert torch.allclose(token_weight, expected_weight)


def test_moe_align_fused_small_token():
    expert_num = 5
    small_topk_ids = torch.tensor([[0, 1, 2], [0, 3, 1], [3, 1, 4]], dtype=torch.int32, device="cuda")
    small_topk_weights = torch.tensor(
        [[0.3, 0.7, 0.1], [0.2, 0.8, 0.4], [0.5, 0.6, 0.9]], dtype=torch.float32, device="cuda"
    )
    _check_moe_align_fused(small_topk_ids, small_topk_weights, expert_num)

    small_many_topk_ids = torch.arange(128 * 17, dtype=torch.int32, device="cuda").reshape(128, 17) % expert_num
    small_many_topk_weights = torch.arange(small_many_topk_ids.numel(), dtype=torch.float32, device="cuda").reshape(
        128, 17
    )
    _check_moe_align_fused(small_many_topk_ids, small_many_topk_weights, expert_num)


def test_moe_align_fused_large_token():
    expert_num = 5

    base_topk_ids = torch.tensor([[0, 1, 2], [0, 3, 1], [3, 1, 4], [2, 0, 4]], dtype=torch.int32, device="cuda")
    large_topk_ids = base_topk_ids.repeat(33, 1)[:129].contiguous()
    large_topk_weights = torch.arange(large_topk_ids.numel(), dtype=torch.float32, device="cuda").reshape(129, 3)
    _check_moe_align_fused(large_topk_ids, large_topk_weights, expert_num, ordered=False)

    medium_topk_ids = base_topk_ids.repeat(1024, 1).contiguous()
    medium_topk_weights = torch.arange(medium_topk_ids.numel(), dtype=torch.float32, device="cuda").reshape(4096, 3)
    _check_moe_align_fused(medium_topk_ids, medium_topk_weights, expert_num, ordered=False)

    shared_expert_num = 257
    shared_routing = torch.arange(512 * 7, dtype=torch.int32, device="cuda").reshape(512, 7) % 256
    shared_last = torch.full((512, 1), 256, dtype=torch.int32, device="cuda")
    shared_topk_ids = torch.cat([shared_routing, shared_last], dim=1).contiguous()
    shared_topk_weights = torch.arange(shared_topk_ids.numel(), dtype=torch.float32, device="cuda").reshape(512, 8)
    _check_moe_align_fused(shared_topk_ids, shared_topk_weights, shared_expert_num, ordered=False)

    large_atomic_topk_ids = base_topk_ids.repeat(1281, 1)[:5121].contiguous()
    large_atomic_topk_weights = torch.arange(large_atomic_topk_ids.numel(), dtype=torch.float32, device="cuda").reshape(
        5121, 3
    )
    _check_moe_align_fused(large_atomic_topk_ids, large_atomic_topk_weights, expert_num, ordered=False)

    sparse_expert_num = 257
    sparse_topk_ids = base_topk_ids.repeat(1281, 1)[:5121].contiguous()
    sparse_topk_weights = torch.arange(sparse_topk_ids.numel(), dtype=torch.float32, device="cuda").reshape(5121, 3)
    _check_moe_align_fused(sparse_topk_ids, sparse_topk_weights, sparse_expert_num, ordered=False)


def test_moe_align_fused_large_token_unordered():
    expert_num = 257
    topk_ids = torch.arange(5121 * 8, dtype=torch.int32, device="cuda").reshape(5121, 8) % expert_num
    topk_weights = torch.arange(topk_ids.numel(), dtype=torch.float32, device="cuda").reshape(5121, 8)
    _check_moe_align_fused(topk_ids, topk_weights, expert_num, ordered=False)


def test_moe_align_fused_atomic_token_unordered():
    expert_num = 9
    topk_ids = torch.arange(257 * 4, dtype=torch.int32, device="cuda").reshape(257, 4) % expert_num
    topk_weights = torch.arange(topk_ids.numel(), dtype=torch.float32, device="cuda").reshape(257, 4)
    expert_to_token_index = torch.empty((expert_num, topk_ids.numel()), dtype=torch.int32, device="cuda")
    expert_to_weight = torch.empty((expert_num, topk_ids.numel()), dtype=torch.float32, device="cuda")
    expert_token_num = torch.empty((expert_num,), dtype=torch.int32, device="cuda")

    _moe_align_fused_atomic_token(
        expert_to_token_index,
        expert_to_weight,
        expert_token_num,
        topk_ids,
        topk_weights,
    )
    torch.cuda.synchronize()

    flat_topk_ids = topk_ids.flatten()
    flat_topk_weights = topk_weights.flatten()
    expected_token_num = torch.bincount(flat_topk_ids, minlength=expert_num).to(torch.int32)
    assert torch.equal(expert_token_num, expected_token_num)

    for expert_id, token_num in enumerate(expected_token_num.tolist()):
        expected_index = torch.nonzero(flat_topk_ids == expert_id, as_tuple=False).flatten().to(torch.int32)
        expected_weight = flat_topk_weights[expected_index]
        token_index = expert_to_token_index[expert_id, :token_num]
        token_weight = expert_to_weight[expert_id, :token_num]
        order = torch.argsort(token_index)
        assert torch.equal(token_index[order], expected_index)
        assert torch.allclose(token_weight[order], expected_weight)


def test_fused_experts_atomic_align_path_is_deterministic():
    token_num = 129
    expert_num = 9
    hidden_size = 64
    intermediate_size = 128
    topk = 4
    hidden_states = torch.randn((token_num, hidden_size), dtype=torch.bfloat16, device="cuda") / 10
    w1 = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16, device="cuda") / 10
    w2 = torch.randn((expert_num, hidden_size, intermediate_size // 2), dtype=torch.bfloat16, device="cuda") / 10
    topk_ids = torch.arange(token_num * topk, dtype=torch.int32, device="cuda").reshape(token_num, topk) % expert_num
    topk_weights = torch.softmax(torch.randn((token_num, topk), dtype=torch.float32, device="cuda"), dim=-1)

    out_0 = fused_experts_impl(hidden_states, w1, w2, topk_weights, topk_ids)
    out_1 = fused_experts_impl(hidden_states, w1, w2, topk_weights, topk_ids)
    torch.cuda.synchronize()

    assert torch.equal(out_0, out_1)


def test_moe_align2():

    experts_token_num = torch.zeros((4,), dtype=torch.int32, device="cuda")
    experts_token_num[0] = 8
    experts_token_num[1] = 0
    experts_token_num[2] = 60
    experts_token_num[3] = 16

    mblocks_to_tuple_info = moe_align2(100, experts_token_num, block_m=16)
    expected_expert_ids = torch.tensor([0, 2, 2, 2, 2, 3, -1, -1, -1, -1], device="cuda", dtype=torch.int32)
    valid_blocks = expected_expert_ids != -1

    assert mblocks_to_tuple_info.shape[0] == triton.cdiv(100 + 4 * (16 - 1), 16)
    assert torch.equal(mblocks_to_tuple_info[:, 0], expected_expert_ids)
    assert torch.equal(
        mblocks_to_tuple_info[valid_blocks, 1],
        torch.tensor([0, 0, 1, 2, 3, 0], device="cuda", dtype=torch.int32),
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
