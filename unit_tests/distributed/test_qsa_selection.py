import datetime
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, world, rendezvous):
    torch.cuda.set_device(rank)
    os.environ["LIGHTLLM_DP_WORLD_SIZE"] = str(world)
    dist.init_process_group(
        "nccl",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=world,
        timeout=datetime.timedelta(seconds=180),
    )
    from lightllm.common.qsa.indexer import QSAIndexer
    from lightllm.models.qwen4_exp.state import QSAExecutionState

    graph = None
    try:
        torch.manual_seed(3519)
        indexer = QSAIndexer(
            torch.randn(640, 32, dtype=torch.bfloat16, device="cuda"),
            torch.zeros(128, dtype=torch.bfloat16, device="cuda"),
            torch.zeros(128, dtype=torch.bfloat16, device="cuda"),
            4,
            2048,
            4,
            use_cuda_topk=True,
        )
        for compact in (False, True):
            capacity = 32768
            cache = torch.randn(capacity // 4 if compact else capacity, 128, device="cuda").to(torch.float8_e4m3fn)
            pages = torch.randperm(capacity // 4, device="cuda").reshape(4, -1)
            table = (pages[:, :, None] * 4 + torch.arange(4, device="cuda")).reshape(4, -1).int()
            view = QSAExecutionState(None, cache, None, None, torch.empty(0) if compact else None)
            for tokens, decode in (
                (world, True),
                (2 * world, True),
                (3 * world, True),
                (4 * world, True),
                (8 * world, True),
                (4 * world + 1, True),
                (1024, False),
            ):
                query = torch.randn(tokens, 4, 128, device="cuda").to(cache.dtype)
                batch = SimpleNamespace(
                    query_rows=(torch.arange(tokens, device="cuda") // 4 % 4).int(),
                    visible_lengths=torch.full((tokens,), table.shape[1], device="cuda", dtype=torch.int32),
                    slot_table=table,
                )

                def reference():
                    return view.select(indexer, query, batch, decode, None, 1, 0, None)

                def run():
                    return view.select(indexer, query, batch, decode, None, world, rank, dist.group.WORLD)

                expected = reference()
                torch.testing.assert_close(run(), expected, atol=0, rtol=0)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        run()
                torch.cuda.current_stream().wait_stream(stream)
                dist.barrier(device_ids=[rank])
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = run()
                for step in range(2):
                    query.copy_(torch.randn(query.shape, device="cuda").to(query.dtype))
                    if step == 1:
                        query.zero_()
                    batch.visible_lengths.copy_(
                        torch.arange(tokens, device="cuda").mul_(137).add_(step * 19).remainder_(table.shape[1])
                    )
                    batch.query_rows.add_(1).remainder_(4)
                    expected = reference()
                    graph.replay()
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                graph.reset()
    finally:
        if graph is not None:
            graph.reset()
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 8])
def test_tp_sharded_qsa_selection_matches_replicated_with_graph(world, tmp_path):
    if torch.cuda.device_count() < world:
        pytest.skip(f"Requires {world} CUDA GPUs")
    mp.spawn(_worker, args=(world, str(tmp_path / "qsa-rendezvous")), nprocs=world, join=True)
