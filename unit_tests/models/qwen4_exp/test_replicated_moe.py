import pytest
import torch
import triton
import torch.nn.functional as F

from lightllm.models.qwen3next.triton_kernel.shared_expert_gate import sigmoid_mul_
from lightllm.models.qwen4_exp.triton_kernel.replicated_moe import (
    expert_down,
    expert_up,
    pack_routes,
    sum_partials,
    sum_routes,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("rows", [4, 28, 64])
def test_routes_graph(rows):
    topk, experts, first = 10, 64, 192
    ids = torch.zeros(rows, topk, dtype=torch.int32, device="cuda")
    mask = torch.ones(rows, dtype=torch.bool, device="cuda")
    masked = torch.empty_like(ids)
    routes = torch.empty(experts, rows * topk, dtype=torch.int32, device="cuda")
    counts = torch.empty(experts, dtype=torch.int32, device="cuda")

    def run():
        pack_routes[(experts,)](
            ids,
            mask,
            masked,
            routes,
            counts,
            rows,
            topk,
            first,
            triton.next_power_of_2(rows * topk),
            num_warps=4 if rows * topk > 256 else 1,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for change in range(4):
        torch.manual_seed(947 + change)
        ids.copy_(torch.rand(rows, 512, device="cuda").argsort(dim=1)[:, :topk])
        if change == 2:
            ids.copy_(first + torch.arange(topk, device="cuda")[None, :])
        mask.copy_(torch.arange(rows, device="cuda") % 3 != 0)
        if change == 0:
            mask.zero_()
        graph.replay()
        expected = ids.masked_fill(~mask[:, None], -1)
        torch.testing.assert_close(masked, expected, rtol=0, atol=0)
        for expert in range(experts):
            positions = (expected.flatten() == first + expert).nonzero().flatten().to(torch.int32)
            assert counts[expert].item() == positions.numel()
            torch.testing.assert_close(routes[expert, : positions.numel()], positions, rtol=0, atol=0)
    graph.reset()


def test_partials_order_graph():
    rows, hidden, topk = 3, 513, 10
    ids = torch.full((rows, topk), -1, dtype=torch.int32, device="cuda")
    partials = torch.zeros(8, rows, hidden, dtype=torch.bfloat16, device="cuda")
    partials[0].fill_(2 ** 25)
    partials[1].fill_(-(2 ** 25))
    partials[2].fill_(1)
    output = torch.empty(rows, hidden, dtype=torch.bfloat16, device="cuda")

    def run():
        sum_partials[(rows, triton.cdiv(hidden, 512))](
            partials,
            ids,
            output,
            hidden,
            rows,
            topk,
            64,
            512,
            10,
            num_warps=4,
            enable_fp_fusion=False,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for routes, expected in (([0, 64, 128], 1), ([0, 64, 128, 0], 0), ([], 0)):
        ids.fill_(-1)
        if routes:
            ids[:, : len(routes)] = torch.tensor(routes, device="cuda", dtype=ids.dtype)
        graph.replay()
        torch.testing.assert_close(output, torch.full_like(output, expected), rtol=0, atol=0)
    graph.reset()


def test_experts_graph():
    torch.manual_seed(1337)
    rows, experts, hidden, intermediate, topk = 28, 4, 2560, 640, 3
    inputs = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    up = torch.randn(experts, 2 * intermediate, hidden, dtype=inputs.dtype, device="cuda") * 0.02
    down = torch.randn(experts, hidden, intermediate, dtype=inputs.dtype, device="cuda") * 0.02
    ids = torch.zeros(rows, topk, dtype=torch.int32, device="cuda")
    weights = torch.full((rows, topk), 1 / topk, device="cuda")
    mask = torch.ones(rows, dtype=torch.bool, device="cuda")
    masked = torch.empty_like(ids)
    routes = torch.empty(experts, rows * topk, dtype=torch.int32, device="cuda")
    counts = torch.empty(experts, dtype=torch.int32, device="cuda")
    mid = torch.empty(rows * topk, intermediate, dtype=inputs.dtype, device="cuda")
    projected = torch.empty(rows * topk, hidden, dtype=inputs.dtype, device="cuda")
    output = torch.empty_like(inputs)

    def run():
        pack_routes[(experts,)](
            ids, mask, masked, routes, counts, rows, topk, 0, triton.next_power_of_2(rows * topk), num_warps=1
        )
        expert_up[(experts, triton.cdiv(intermediate, 32))](
            inputs,
            up,
            mid,
            routes,
            counts,
            hidden,
            intermediate,
            rows,
            topk,
            16,
            32,
            256,
            num_warps=4,
            num_stages=3,
        )
        expert_down[(experts, triton.cdiv(hidden, 64))](
            mid,
            down,
            projected,
            routes,
            counts,
            intermediate,
            hidden,
            rows,
            topk,
            1,
            16,
            64,
            128,
            num_warps=4,
            num_stages=3,
        )
        sum_routes[(rows, triton.cdiv(hidden, 256))](
            projected, weights, masked, output, hidden, topk, 0, experts, triton.next_power_of_2(topk), num_warps=4
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for change in range(4):
        inputs.mul_(0.9)
        ids.copy_(torch.rand(rows, experts, device="cuda").argsort(dim=1)[:, :topk])
        if change == 2:
            ids.copy_(torch.arange(topk, device="cuda")[None, :])
        mask.copy_(torch.arange(rows, device="cuda") % 3 != 0)
        if change == 0:
            mask.zero_()
        graph.replay()
        expected = torch.zeros(rows, hidden, dtype=torch.float32, device="cuda")
        for route in range(topk):
            for expert in range(experts):
                active = mask & (ids[:, route] == expert)
                gate, value = F.linear(inputs[active], up[expert]).chunk(2, dim=-1)
                activated = F.silu(gate.float()).to(inputs.dtype) * value
                result = F.linear(activated, down[expert]).float()
                expected[active] += result * weights[active, route, None]
        torch.testing.assert_close(output, expected.to(output.dtype), rtol=0.008, atol=0.002)
        assert torch.count_nonzero(output[~mask]) == 0
    graph.reset()


@pytest.mark.parametrize("rows,hidden", [(4, 513), (28, 2560), (64, 2560)])
def test_shared_gate_and_partials_graph(rows, hidden):
    torch.manual_seed(7911)
    ids = torch.zeros(rows, 10, dtype=torch.int32, device="cuda")
    partials = torch.randn(8, rows, hidden, dtype=torch.bfloat16, device="cuda")
    shared = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    gate = torch.empty(rows, 1, dtype=torch.bfloat16, device="cuda")
    actual = torch.empty_like(shared)
    expected = torch.empty_like(shared)

    def reduce(output, **kwargs):
        sum_partials[(rows, triton.cdiv(hidden, 512))](
            partials,
            ids,
            output,
            hidden,
            rows,
            10,
            64,
            512,
            10,
            num_warps=4,
            enable_fp_fusion=False,
            **kwargs,
        )

    gate.zero_()
    reduce(actual, Shared=shared, Gate=gate)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        reduce(actual, Shared=shared, Gate=gate)
    for change in range(8):
        partials.normal_()
        shared.normal_()
        gate.copy_(torch.linspace(-12, 12, rows, device="cuda")[:, None])
        ids.copy_(torch.rand(rows, 512, device="cuda").argsort(dim=1)[:, :10])
        ids[::3].fill_(-1)
        if change % 2:
            partials[0].fill_(2 ** 25)
            partials[1].fill_(-(2 ** 25))
            partials[2].fill_(1)
            ids[1] = torch.tensor([0, 64, 128, 0, -1, -1, -1, -1, -1, -1], device="cuda")
        reduce(expected)
        expected.add_(sigmoid_mul_(shared.clone(), gate))
        graph.replay()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    graph.reset()
