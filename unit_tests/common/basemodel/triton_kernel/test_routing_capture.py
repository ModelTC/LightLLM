import pytest
import torch

from lightllm.common.basemodel.triton_kernel.routing_capture import scatter_routing_topk_to_cpu


def _skip_without_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton kernels.")


@pytest.mark.parametrize("dtype,dtype_id,max_value", [(torch.uint8, 1, 255), (torch.int16, 2, 1024)])
def test_scatter_routing_topk_to_pinned_cpu(dtype, dtype_id, max_value):
    _skip_without_cuda()

    num_tokens = 5
    num_moe_layers = 3
    topk = 4
    kv_cache_size = 32
    moe_layer_index = 1

    topk_ids = torch.randint(0, max_value, (num_tokens, topk), dtype=torch.int64, device="cuda")
    mem_indexes = torch.tensor([17, 3, 29, 8, 11], dtype=torch.int32, device="cuda")
    routing_buffer = torch.zeros(
        (kv_cache_size, num_moe_layers, topk),
        dtype=dtype,
        device="cpu",
        pin_memory=True,
    )
    routing_buffer_ptr = torch.tensor([routing_buffer.data_ptr()], dtype=torch.uint64, device="cuda")

    scatter_routing_topk_to_cpu(
        topk_ids=topk_ids,
        mem_indexes=mem_indexes,
        routing_buffer_ptr=routing_buffer_ptr,
        moe_layer_index=moe_layer_index,
        num_moe_layers=num_moe_layers,
        topk=topk,
        dtype_id=dtype_id,
    )
    torch.cuda.synchronize()

    expected = torch.zeros_like(routing_buffer)
    expected[mem_indexes.cpu().long(), moe_layer_index, :] = topk_ids.cpu().to(dtype)
    assert torch.equal(routing_buffer, expected)


def test_scatter_routing_topk_respects_layer_index():
    _skip_without_cuda()

    num_tokens = 3
    num_moe_layers = 2
    topk = 2
    kv_cache_size = 16

    topk_ids = torch.arange(num_tokens * topk, dtype=torch.int64, device="cuda").view(num_tokens, topk)
    mem_indexes = torch.tensor([10, 4, 13], dtype=torch.int64, device="cuda")
    routing_buffer = torch.zeros(
        (kv_cache_size, num_moe_layers, topk),
        dtype=torch.int16,
        device="cpu",
        pin_memory=True,
    )
    routing_buffer_ptr = torch.tensor([routing_buffer.data_ptr()], dtype=torch.uint64, device="cuda")

    scatter_routing_topk_to_cpu(
        topk_ids=topk_ids,
        mem_indexes=mem_indexes,
        routing_buffer_ptr=routing_buffer_ptr,
        moe_layer_index=1,
        num_moe_layers=num_moe_layers,
        topk=topk,
        dtype_id=2,
    )
    torch.cuda.synchronize()

    expected = torch.zeros_like(routing_buffer)
    expected[mem_indexes.cpu(), 1, :] = topk_ids.cpu().to(torch.int16)
    assert torch.equal(routing_buffer, expected)


def test_scatter_routing_topk_is_cuda_graph_capturable():
    _skip_without_cuda()

    num_tokens = 4
    num_moe_layers = 2
    topk = 3
    kv_cache_size = 16

    topk_ids = torch.arange(num_tokens * topk, dtype=torch.int64, device="cuda").view(num_tokens, topk)
    mem_indexes = torch.tensor([2, 4, 6, 8], dtype=torch.int32, device="cuda")
    routing_buffer = torch.zeros(
        (kv_cache_size, num_moe_layers, topk),
        dtype=torch.uint8,
        device="cpu",
        pin_memory=True,
    )
    routing_buffer_ptr = torch.tensor([routing_buffer.data_ptr()], dtype=torch.uint64, device="cuda")

    scatter_routing_topk_to_cpu(
        topk_ids=topk_ids,
        mem_indexes=mem_indexes,
        routing_buffer_ptr=routing_buffer_ptr,
        moe_layer_index=0,
        num_moe_layers=num_moe_layers,
        topk=topk,
        dtype_id=1,
    )
    torch.cuda.synchronize()
    routing_buffer.zero_()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scatter_routing_topk_to_cpu(
            topk_ids=topk_ids,
            mem_indexes=mem_indexes,
            routing_buffer_ptr=routing_buffer_ptr,
            moe_layer_index=0,
            num_moe_layers=num_moe_layers,
            topk=topk,
            dtype_id=1,
        )

    graph.replay()
    torch.cuda.synchronize()

    expected = torch.zeros_like(routing_buffer)
    expected[mem_indexes.cpu(), 0, :] = topk_ids.cpu().to(torch.uint8)
    assert torch.equal(routing_buffer, expected)


if __name__ == "__main__":
    pytest.main([__file__])
