import importlib
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.pinned_embedding_weight import PinnedRowShardedEmbeddingWeight
from lightllm.models.qwen4_exp import pinned_memory

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    base = importlib.import_module("lightllm.common.basemodel.layer_weights.meta_weights.base_weight")
    platform = importlib.import_module("lightllm.common.basemodel.layer_weights.meta_weights.platform_op")
    monkeypatch.setattr(base, "get_current_device_id", torch.cuda.current_device)
    monkeypatch.setattr(
        platform,
        "get_env_start_args",
        lambda: SimpleNamespace(hardware_platform="cuda", enable_torch_fallback=False, enable_triton_fallback=False),
    )
    monkeypatch.setattr(pinned_memory, "_LARGE_BYTES", 1)
    monkeypatch.setattr(pinned_memory, "_CHUNK_BYTES", 65536)


def weight(dtype, rank=0, world=1):
    return PinnedRowShardedEmbeddingWeight(
        160, 1024, "table.weight", dtype, 4, cpu_allocator=pinned_memory.pinned_zeros, tp_rank=rank, tp_world_size=world
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_streamed_host_shards_and_vocab_parallel_gather(dtype):
    table = (torch.arange(1024 * 160).reshape(1024, 160) % 127 - 64).to(dtype)
    ids = torch.tensor([0, 255, 256, 340, 341, 511, 512, 681, 682, 1023, -1, 1024], device="cuda")
    outputs = []
    for rank in range(3):
        value = weight(dtype, rank, 3)
        address = value.weight.data_ptr()
        assert not value.verify_load() and value.weight.device.type == "cpu" and value.weight.is_pinned()
        assert len(value.weight._qwen4_pinned_allocation.addresses) > 1
        for index in (2, 0, 3, 1):
            value.load_hf_weights({f"table.shard_{index}.weight": table[index * 256 : (index + 1) * 256]})
        assert value.verify_load() and value.weight.data_ptr() == address
        torch.testing.assert_close(value.weight, table[value.tp_vocab_start_id : value.tp_vocab_end_id], atol=0, rtol=0)
        output = value(ids)
        assert output.is_cuda
        outputs.append(output)
    expected = table.cuda().index_select(0, ids.clamp(0, 1023))
    expected[(ids < 0) | (ids >= 1024)] = 0
    torch.testing.assert_close(torch.stack(outputs).sum(0), expected, atol=0, rtol=0)


def test_device_checkpoint_staging_and_dynamic_graph_replay():
    table = (torch.arange(1024 * 160).reshape(1024, 160) % 127).to(torch.bfloat16).cuda()
    value = weight(torch.bfloat16)
    value.load_hf_weights({"table.weight": table})
    assert value.verify_load()
    ids = torch.tensor([0, 191, 192, 511, 512, 1023], device="cuda")
    output = torch.empty((ids.numel(), 160), dtype=table.dtype, device="cuda")
    value(ids, out=output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        value(ids, out=output)
    ids.copy_(torch.tensor([1023, 512, 511, 192, 191, 0], device="cuda"))
    graph.replay()
    torch.testing.assert_close(output, table.index_select(0, ids), atol=0, rtol=0)
    graph.reset()


def test_rejects_pageable_allocator_and_wrong_output():
    with pytest.raises(ValueError, match="pinned CPU"):
        PinnedRowShardedEmbeddingWeight(
            160,
            1024,
            "table.weight",
            torch.bfloat16,
            4,
            cpu_allocator=lambda shape, **kwargs: torch.zeros(shape, dtype=kwargs["dtype"]),
            tp_rank=0,
            tp_world_size=1,
        )
    value = weight(torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA vector"):
        value(torch.tensor([0]))
    with pytest.raises(ValueError, match="output"):
        value(torch.tensor([0], device="cuda"), out=torch.empty((1, 160), device="cuda", dtype=torch.float32))
