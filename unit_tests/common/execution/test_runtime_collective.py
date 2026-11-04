from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lightllm.distributed import symm_mem_all_reduce as module


@pytest.mark.parametrize("failure", [NotImplementedError, RuntimeError])
def test_collective_dtype_is_checked_before_enabling(failure, monkeypatch):
    calls, synchronized = [], []
    monkeypatch.setattr(module, "_SYMM_MEM_OK", True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(module.torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(module.torch.cuda, "synchronize", synchronized.append)
    monkeypatch.setattr(
        module,
        "torch_symm_mem",
        SimpleNamespace(
            empty=lambda size, device, dtype: torch.empty(32, dtype=dtype),
            rendezvous=lambda buffer, group: SimpleNamespace(multicast_ptr=1),
        ),
    )

    def unsupported(tensor, operation, group):
        calls.append((tensor.dtype, operation, group))
        raise failure("unsupported collective dtype" if failure is NotImplementedError else "device failure")

    monkeypatch.setattr(module.torch.ops.symm_mem, "two_shot_all_reduce_", unsupported, raising=False)
    if failure is NotImplementedError:
        bound = module.SymmMemAllreduce(SimpleNamespace(group_name="tp"), 0, dtype=torch.float16)
        assert bound.disabled and bound.buffer is None
        assert not bound.should_use(torch.empty(32, dtype=torch.float16))
    else:
        with pytest.raises(RuntimeError, match="device failure"):
            module.SymmMemAllreduce(SimpleNamespace(group_name="tp"), 0, dtype=torch.float16)
    assert calls == [(torch.float16, "sum", "tp")]
    assert synchronized == ([torch.device("cuda:0")] if failure is NotImplementedError else [])


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_two_rank_collective_and_graph_with_model_dtype(dtype, tmp_path):
    context = mp.spawn(_collective_worker, args=(str(tmp_path / "init"), dtype), nprocs=2, join=False)
    try:
        for _ in range(60):
            if context.join(timeout=5):
                return
        raise TimeoutError("two-rank collective validation did not finish")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)


def _collective_worker(rank, init_file, dtype):
    graph = None
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method="file://" + init_file,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
        device_id=torch.device("cuda", rank),
    )
    try:
        group = dist.distributed_c10d._get_default_group()
        bound = module.SymmMemAllreduce(group, rank, dtype=dtype)
        if dtype == torch.bfloat16:
            assert not bound.disabled
        inputs = torch.full((8, 128), rank + 1, dtype=dtype, device="cuda")

        def reduce():
            if bound.should_use(inputs):
                bound.all_reduce(inputs)
            else:
                dist.all_reduce(inputs, group=group)

        reduce()
        torch.testing.assert_close(inputs, torch.full_like(inputs, 3), rtol=0, atol=0)
        inputs.fill_(rank + 2)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            reduce()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(inputs, torch.full_like(inputs, 5), rtol=0, atol=0)
        inputs.fill_(rank + 4)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(inputs, torch.full_like(inputs, 9), rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        # NCCL destruction waits for captured graphs to release their communicator.
        if graph is not None:
            graph.reset()
        dist.destroy_process_group()
