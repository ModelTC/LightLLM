import pytest
import torch

from lightllm.common.basemodel.triton_kernel.routing_capture import scatter_routing_capture_to_cpu


@pytest.mark.parametrize("dtype,max_value", [(torch.uint8, 255), (torch.int16, 1024)])
def test_scatter_routing_capture_to_pinned_cpu(dtype, max_value):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton kernels.")

    num_tokens = 5
    num_moe_layers = 3
    topk = 4
    kv_cache_size = 32

    capture_buffer = torch.randint(
        0,
        max_value,
        (num_tokens, num_moe_layers, topk),
        dtype=dtype,
        device="cuda",
    )
    mem_indexes = torch.tensor([17, 3, 29, 8, 11], dtype=torch.int32, device="cuda")
    routing_buffer = torch.zeros(
        (kv_cache_size, num_moe_layers, topk),
        dtype=dtype,
        device="cpu",
        pin_memory=True,
    )

    scatter_routing_capture_to_cpu(
        capture_buffer=capture_buffer,
        mem_indexes=mem_indexes,
        routing_buffer=routing_buffer,
        num_tokens=num_tokens,
        num_moe_layers=num_moe_layers,
        topk=topk,
    )
    torch.cuda.synchronize()

    expected = torch.zeros_like(routing_buffer)
    expected[mem_indexes.cpu().long()] = capture_buffer.cpu()
    assert torch.equal(routing_buffer, expected)


def test_scatter_routing_capture_respects_num_tokens():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton kernels.")

    num_tokens = 3
    max_capture_tokens = 6
    num_moe_layers = 2
    topk = 2
    kv_cache_size = 16

    capture_buffer = torch.arange(
        max_capture_tokens * num_moe_layers * topk,
        dtype=torch.int16,
        device="cuda",
    ).view(max_capture_tokens, num_moe_layers, topk)
    mem_indexes = torch.tensor([10, 4, 13, 1, 2, 3], dtype=torch.int64, device="cuda")
    routing_buffer = torch.zeros(
        (kv_cache_size, num_moe_layers, topk),
        dtype=torch.int16,
        device="cpu",
        pin_memory=True,
    )

    scatter_routing_capture_to_cpu(
        capture_buffer=capture_buffer[:num_tokens],
        mem_indexes=mem_indexes[:num_tokens],
        routing_buffer=routing_buffer,
        num_tokens=num_tokens,
        num_moe_layers=num_moe_layers,
        topk=topk,
    )
    torch.cuda.synchronize()

    expected = torch.zeros_like(routing_buffer)
    expected[mem_indexes[:num_tokens].cpu()] = capture_buffer[:num_tokens].cpu()
    assert torch.equal(routing_buffer, expected)


if __name__ == "__main__":
    pytest.main([__file__])
