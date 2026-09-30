from datetime import timedelta
from types import SimpleNamespace as NS
from unittest.mock import patch
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _transfer_rank(rank, rendezvous, destination_done):
    from lightllm.server.router.model_infer.mode_backend.dp_backend import dp_shared_kv_trans as transfer

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        node = NS(node_prefix_total_len=2816, small_page_buffer_idx=0)
        cache = NS(
            get_mem_index_value_by_node=lambda node, start, end: torch.arange(start, end).int(),
            get_big_page_ids_by_node=lambda node: [0],
        )
        big = torch.full((1, 4), 19 if rank == 1 else 0, dtype=torch.uint8)
        small = torch.full((1, 4), 37 if rank == 1 else 0, dtype=torch.uint8)
        module = object.__new__(transfer.DPKVSharedMoudle)
        module.dp_rank_in_node = rank
        module.backend = NS(
            args=NS(linear_att_hash_page_size=256, linear_att_page_block_num=8, max_req_total_len=4096),
            node_gloo_group=dist.group.WORLD,
            node_nccl_group=dist.group.WORLD,
            radix_cache=cache,
            model=NS(mem_manager=NS(big_page_buffers=NS(buffer=big))),
            small_page_buffers=NS(buffer=small),
        )
        req = NS(req_id=7, hybrid_len_to_big_page_id={2048: 0})
        task = transfer.TransTask(req, torch.empty(2816, dtype=torch.int32), 1, 1, terminal_small_page_buffer_id=0)
        original_to = torch.Tensor.to

        def host_to(tensor, *args, **kwargs):
            # Only index upload is substituted; checkpoint transport uses real Gloo.
            if kwargs.get("device") == "cuda":
                kwargs["device"] = "cpu"
            return original_to(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "to", host_to):
            module._transfer_dsv4_source_data(
                [task] if rank == 0 else [],
                {7: transfer.PrefixCacheMatch(node, cache)} if rank == 1 else {},
                [[(1, 7, 0, 2816)], []],
            )
        if rank == 0:
            assert torch.equal(task.max_kv_len_mem_indexes, torch.arange(2816).int())
            assert big.eq(19).all() and small.eq(37).all()

        def synchronize():
            if rank == 0:
                time.sleep(0.1)
                destination_done.set()

        # Exercise the production completion fence with a source that has no receive task.
        # Gloo substitutes for NCCL, and the stream stub marks delayed copy completion.
        module._transfer_dsv4_source_data = lambda *args: None
        with patch.object(torch.cuda, "current_stream", lambda: NS(synchronize=synchronize)):
            module.kv_trans_dsv4([], {}, [[], []])
        assert destination_done.is_set()
    finally:
        dist.destroy_process_group()


def test_two_rank_checkpoint_transport_and_source_only_completion_fence(tmp_path):
    destination_done = mp.get_context("spawn").Event()
    mp.spawn(_transfer_rank, args=(str(tmp_path / "rendezvous"), destination_done), nprocs=2, join=True)
