import datetime
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, world, rendezvous):
    torch.cuda.set_device(rank)
    from lightllm.server.core.objs.start_args_type import StartArgs

    settings = StartArgs()
    settings.hardware_platform, settings.data_type = "cuda", "bf16"
    os.environ["LIGHTLLM_START_ARGS"] = json.dumps(vars(settings))
    for name, value in {
        "LIGHTLLM_DP_WORLD_SIZE": world,
        "LIGHTLLM_GLOBAL_WORLD_SIZE": world,
        "LIGHTLLM_NODE_WORLD_SIZE": world,
        "LIGHTLLM_CURRENT_DEVICE_ID": rank,
        "LIGHTLLM_CURRENT_RANK_IN_DP": rank,
        "LIGHTLLM_CURRENT_RANK_IN_NODE": rank,
        "LIGHTLLM_GLOBAL_RANK": rank,
        "LIGHTLLM_DP_SIZE": 1,
        "LIGHTLLM_GLOBAL_DP_RANK": 0,
        "LIGHTLLM_DP_RANK_IN_NODE": 0,
    }.items():
        os.environ[name] = str(value)
    dist.init_process_group(
        "nccl",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=world,
        timeout=datetime.timedelta(seconds=180),
    )
    from lightllm.distributed.communication_op import CustomProcessGroup
    from lightllm.distributed.symm_mem_all_reduce import SymmMemAllreduce
    from lightllm.common.basemodel.tp_reduce import ordered_tp_sum
    from lightllm.distributed.communication_op import all_gather_into_tensor

    graph = None
    try:
        symmetric = SymmMemAllreduce(dist.group.WORLD, rank)
        available = torch.tensor(int(not symmetric.disabled), device=rank)
        dist.all_reduce(available, op=dist.ReduceOp.MIN)
        if not available.item():
            if rank == 0:
                Path(rendezvous + ".unavailable").touch()
            return
        group = CustomProcessGroup.__new__(CustomProcessGroup)
        group.device_group, group.dp_world_size = dist.group.WORLD, world
        group.symm_mem_reduce, group.flashinfer_reduce = symmetric, None
        for rows in (1, 8, 24, 128, 512, 1024, 4096, 8192):
            source = torch.empty((rows, 2560), dtype=torch.bfloat16, device=rank)
            reference = torch.empty((world * rows, 2560), dtype=source.dtype, device=rank)
            scratch = torch.empty_like(source)

            def expected():
                dist.all_gather_into_tensor(reference, source)
                return reference.view(world, rows, 2560).double().sum(0).to(source.dtype)

            for iteration in range(4):
                if world >= 4 and iteration == 0:
                    source.fill_((2 ** 40, 1, -(2 ** 40), 1)[rank] if rank < 4 else 0)
                else:
                    source.copy_((torch.arange(source.numel(), device=rank).reshape_as(source) % 257 + rank) / 64)
                    source.add_(iteration / 8)
                gold = expected()
                gathered = symmetric.ordered_all_gather(source)
                scratch.fill_(rank + 2)
                symmetric.all_reduce(scratch)
                if gathered is not None:
                    torch.testing.assert_close(gathered.view_as(reference), reference, rtol=0, atol=0)
                actual = ordered_tp_sum(source.clone(), group, world)
                torch.testing.assert_close(actual, gold, rtol=0, atol=0)
                owned = torch.empty_like(reference)
                all_gather_into_tensor(owned, source, group=group)
                symmetric.ordered_all_gather(source + 1)
                torch.testing.assert_close(owned, reference, rtol=0, atol=0)
                work = all_gather_into_tensor(owned, source, group=group, async_op=True)
                assert work is not None
                work.wait()
                torch.testing.assert_close(owned, reference, rtol=0, atol=0)
                if rows % world == 0:
                    actual = ordered_tp_sum(source, group, world, scatter_rank=rank)
                    torch.testing.assert_close(actual, gold.chunk(world)[rank], rtol=0, atol=0)
                    actual = ordered_tp_sum(source, dist.group.WORLD, world, scatter_rank=rank)
                    torch.testing.assert_close(actual, gold.chunk(world)[rank], rtol=0, atol=0)

            output = torch.empty_like(source)
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                output.copy_(source)
                ordered_tp_sum(output, group, world)
            for iteration in range(8):
                source.fill_(rank + iteration / 16)
                gold = expected()
                graph.replay()
                torch.testing.assert_close(output, gold, rtol=0, atol=0)
            graph.reset()

            if rows % world == 0:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    scattered = ordered_tp_sum(source, group, world, scatter_rank=rank)
                    symmetric.all_reduce(scratch)
                    symmetric.ordered_all_gather(source[:1])
                for iteration in range(4):
                    source.fill_((2 ** 40, 1, -(2 ** 40), 1)[rank] if rank < 4 else iteration)
                    gold = expected()
                    graph.replay()
                    torch.testing.assert_close(scattered, gold.chunk(world)[rank], rtol=0, atol=0)
                graph.reset()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    scattered = ordered_tp_sum(source, dist.group.WORLD, world, scatter_rank=rank)
                for iteration in range(4):
                    source.fill_((2 ** 40, 1, -(2 ** 40), 1)[rank] if rank < 4 else iteration)
                    gold = expected()
                    graph.replay()
                    torch.testing.assert_close(scattered, gold.chunk(world)[rank], rtol=0, atol=0)
                graph.reset()

            owned = torch.empty_like(reference)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                all_gather_into_tensor(owned, source, group=group)
                symmetric.ordered_all_gather(source + 1)
            for iteration in range(8):
                source.fill_(rank + iteration / 8)
                expected()
                graph.replay()
                torch.testing.assert_close(owned, reference, rtol=0, atol=0)
            graph.reset()
        for dtype in (torch.int32, torch.int64):
            for rows in (4, 128, 1024):
                source = torch.empty(rows, 2051, dtype=dtype, device=rank)
                reference = torch.empty(world * rows, 2051, dtype=dtype, device=rank)
                owned = torch.empty_like(reference)
                patterns = torch.tensor([-1, 0, 1, 0x7FC00000, -2147483648, 2147483647], device=rank, dtype=dtype)
                if dtype == torch.int64:
                    patterns = torch.cat((patterns, torch.tensor([-(2 ** 63), 2 ** 63 - 1], device=rank)))

                def update(iteration):
                    source.copy_(
                        patterns[
                            (torch.arange(source.numel(), device=rank) + rank + iteration) % patterns.numel()
                        ].view_as(source)
                    )
                    dist.all_gather_into_tensor(reference, source)

                update(0)
                all_gather_into_tensor(owned, source, group=group)
                torch.testing.assert_close(owned, reference, rtol=0, atol=0)
                work = all_gather_into_tensor(owned, source, group=group, async_op=True)
                work.wait()
                torch.testing.assert_close(owned, reference, rtol=0, atol=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    all_gather_into_tensor(owned, source, group=group)
                    symmetric.ordered_all_gather(torch.ones(16, dtype=symmetric.dtype, device=rank))
                for iteration in range(3):
                    update(iteration + 1)
                    graph.replay()
                    torch.testing.assert_close(owned, reference, rtol=0, atol=0)
                graph.reset()

    finally:
        if graph is not None:
            graph.reset()
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 8])
def test_ordered_sum_preserves_shards_across_collectives_and_graph_replays(world, tmp_path):
    if torch.cuda.device_count() < world:
        pytest.skip(f"Requires {world} CUDA GPUs")
    if any(torch.cuda.get_device_capability(rank)[0] < 9 for rank in range(world)):
        pytest.skip("Requires Hopper or newer GPUs with multicast support")
    from lightllm.utils.device_utils import has_nvlink

    if not has_nvlink():
        pytest.skip("Requires an NVLink-connected node")
    rendezvous = str(tmp_path / "rendezvous")
    mp.spawn(_worker, args=(world, rendezvous), nprocs=world, join=True)
    if Path(rendezvous + ".unavailable").exists():
        pytest.skip("Symmetric multicast memory is unavailable")
