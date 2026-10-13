from datetime import timedelta
from types import SimpleNamespace as NS
import os
import uuid
import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _rank(rank, rendezvous, service_name):
    os.environ["LIGHTLLM_UNIQUE_SERVICE_NAME_ID"] = service_name
    os.environ["LIGHTLLM_CURRENT_DEVICE_ID"] = str(rank)
    os.environ["LIGHTLLM_CURRENT_RANK_IN_NODE"] = str(rank)
    from lightllm.server.router.model_infer.mode_backend import generic_pre_process  # noqa: F401
    from lightllm.common.kv_cache_mem_manager.allocator import KvCacheAllocator
    from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import DeepseekV4CpuCacheLayout
    from lightllm.common.req_manager.deepseek4 import DeepseekV4ReqManager
    from lightllm.common.state_cache_manager.deepseek4 import DeepseekV4StateCacheManager
    from lightllm.server.router.model_infer.mode_backend.dp_backend.dp_shared_kv_trans import (
        DPKVSharedMoudle,
        TransTask,
    )
    from lightllm.utils import dist_utils, envs_utils
    from lightllm.utils.service_shm_cleanup import ServiceShmCleanup

    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        args = NS(dp=2, nnodes=1, linear_att_hash_page_size=256, linear_att_page_block_num=8, max_req_total_len=4096)
        envs_utils.get_env_start_args = lambda: args
        dist_utils.create_new_group_for_current_node = lambda backend: dist.group.WORLD
        dist_utils.get_dp_rank_in_node = lambda: rank
        dist_utils.get_current_rank_in_dp = lambda: 0
        layout = DeepseekV4CpuCacheLayout.from_compress_rates([4, 128])
        big = DeepseekV4StateCacheManager.create_dp_shared("dp_big_checkpoints", 2, layout=layout)
        small = DeepseekV4StateCacheManager.create_dp_shared("dp_small_checkpoints", 2, layout=layout)
        assert big.size == small.size == 4
        assert big.buffer.is_pinned() and small.buffer.is_pinned()

        memory = NS(
            swa_pool=NS(buffer=torch.zeros((2, 32, layout.swa_gpu_page_nbytes), dtype=torch.uint8, device="cuda")),
            swa_page_allocator=KvCacheAllocator(32, shared_name=f"test_swa_{rank}"),
            n_c4=1,
            c4_state_ring=8,
            c4_state_buffer=torch.zeros((1, 32 * 8 + 1, 4 * 512), dtype=torch.float32, device="cuda"),
            c4_indexer_state_buffer=torch.zeros((1, 32 * 8 + 1, 4 * 128), dtype=torch.float32, device="cuda"),
            big_page_buffers=big,
        )
        req_manager = object.__new__(DeepseekV4ReqManager)
        req_manager.mem_manager = memory
        req_manager.sliding_window = 256
        req_manager.HOLD_REQUEST_ID = 1
        req_manager._swa_pages = [{}]
        req_manager.req_to_swa_pages = torch.full((1, 32), -1, dtype=torch.int32, device="cuda")
        req_manager.small_page_buffers = small
        checkpoint_ids = []
        if rank == 1:
            req_manager.prepare_swa(0, 0, 2304)
            memory.swa_pool.buffer.fill_(19)
            memory.c4_state_buffer.fill_(23)
            memory.c4_indexer_state_buffer.fill_(29)
            big_id, small_id = big.alloc_one_state_cache(), small.alloc_one_state_cache()
            req_manager.save_state(0, big_id, big, 2048)
            req_manager.save_state(0, small_id, small, 2304)
            torch.cuda.synchronize()
            checkpoint_ids = [big_id, small_id]
        dist.barrier()

        module = object.__new__(DPKVSharedMoudle)
        cache = NS(
            get_mem_index_value_by_node=lambda node, start, end: torch.arange(start, end),
            get_big_page_ids_by_node=lambda node: [checkpoint_ids[0]],
        )
        module.backend = NS(
            args=args,
            node_gloo_group=dist.group.WORLD,
            model=NS(mem_manager=memory),
            radix_cache=cache,
            small_page_buffers=small,
        )
        req = NS(
            req_id=7, req_idx=0, hybrid_cache_len=2304, hybrid_len_to_big_page_id={}, tail_small_page_buffer_id=None
        )
        task = TransTask(req, torch.arange(2304, device="cuda"), 1, 1)
        node = NS(small_page_buffer_idx=checkpoint_ids[-1]) if rank == 1 else None
        original_send = dist.send
        transmitted = []

        def record_send(tensor, **kwargs):
            assert tensor.dtype == torch.int64
            transmitted.append(tensor.numel() * tensor.element_size())
            return original_send(tensor, **kwargs)

        dist.send = record_send
        module._transfer_dsv4_source_data(
            [task] if rank == 0 else [],
            {7: NS(node=node, matched_len=2304)} if rank == 1 else {},
            [[(1, 7, 0, 2304)], []],
        )
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 1:
            assert transmitted == [2304 * 8, 2 * 8]
            assert big.refs[big_id] == small.refs[small_id] == 2
            big.free_state_cache([big_id])
            small.free_state_cache([small_id])
            for pool, held_id in [(big, big_id), (small, small_id)]:
                other = pool.alloc_state_cache(3)
                assert held_id not in other
                pool.free_state_cache(other)
        dist.barrier()
        if rank == 0:
            assert task.max_kv_len_mem_indexes.is_cuda
            assert torch.equal(task.max_kv_len_mem_indexes.cpu(), torch.arange(2304, dtype=torch.int32))
            ids = [req.hybrid_len_to_big_page_id[2048], req.tail_small_page_buffer_id]
            assert big.refs[ids[0]] == small.refs[ids[1]] == 1
            for pool, checkpoint_id, length in [(big, ids[0], 2048), (small, ids[1], 2304)]:
                req_manager.restore_state(req, pool, checkpoint_id, length)
                torch.cuda.synchronize()
                pages = req_manager.req_to_swa_pages[0, length // 128 - 2 : length // 128].long()
                assert torch.all(memory.swa_pool.buffer.index_select(1, pages) == 19)
                slots = pages[-1] * 128 + torch.arange(124, 128, device="cuda")
                rows = (slots // 128) * 8 + slots % 8
                assert torch.all(memory.c4_state_buffer.index_select(1, rows) == 23)
                assert torch.all(memory.c4_indexer_state_buffer.index_select(1, rows) == 29)
                pool.free_state_cache([checkpoint_id])
        dist.barrier()
        assert big.get_free_cache_num() == small.get_free_cache_num() == 4
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()
        if rank == 0:
            ServiceShmCleanup(service_name).cleanup_service_resources()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_dsv4_fetch_uses_shared_ids_and_restores_after_source_eviction():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        rendezvous = f"tcp://127.0.0.1:{sock.getsockname()[1]}"
    mp.spawn(_rank, args=(rendezvous, f"test_dsv4_shared_{uuid.uuid4().hex}"), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="requires a CUDA device")
def test_shared_checkpoint_copy_across_default_registration_boundary(monkeypatch):
    from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import DeepseekV4CpuCacheLayout
    from lightllm.common.state_cache_manager.deepseek4 import DeepseekV4StateCacheManager
    from lightllm.utils.service_shm_cleanup import ServiceShmCleanup

    service_name = f"test_dsv4_registration_{uuid.uuid4().hex}"
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", service_name)
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    torch.cuda.set_device(0)
    layout = DeepseekV4CpuCacheLayout.from_compress_rates([0] * 5 + [4] * 21 + [128] * 20)
    state_bytes = layout.page_nbytes - layout.swa_offset
    crossing_index = (128 * 1024 * 1024) // state_bytes
    assert crossing_index * state_bytes < 128 * 1024 * 1024 < (crossing_index + 1) * state_bytes
    try:
        pool = DeepseekV4StateCacheManager(crossing_index + 2, layout, shm_name="checkpoints", shm_create=True)
        source = torch.full((state_bytes,), 173, dtype=torch.uint8, device="cuda")
        pool.buffer[crossing_index].copy_(source, non_blocking=True)
        torch.cuda.synchronize()
        restored = pool.buffer[crossing_index].cuda(non_blocking=True)
        torch.cuda.synchronize()
        assert torch.equal(restored, source)
    finally:
        ServiceShmCleanup(service_name).cleanup_service_resources()
