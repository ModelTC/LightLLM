"""Real TP2 completion and checkpoint reuse; requires two CUDA devices."""

import os
import json
import socket
from datetime import timedelta
from dataclasses import asdict
from collections import deque
from types import SimpleNamespace as NS
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, port):
    torch.cuda.set_device(rank)
    os.environ["LIGHTLLM_CURRENT_DEVICE_ID"] = str(rank)
    os.environ["LIGHTLLM_CURRENT_RANK_IN_NODE"] = str(rank)
    from lightllm.server.core.objs.start_args_type import StartArgs

    os.environ["LIGHTLLM_START_ARGS"] = json.dumps(asdict(StartArgs(cpu_cache_token_page_size=8)))
    from unit_tests.common.att_resources.test_async_cpu_cache import Client
    from lightllm.common.attention_state import AttentionStateManager
    from lightllm.common.attention_state.full_att import GQAFullAttentionState, GQAAttentionConfig
    from lightllm.server.router.model_infer.mode_backend.multi_level_kv_cache import (
        MultiLevelKvCacheModule,
        CacheSession,
        StagingSlot,
    )
    from lightllm.server.router.model_infer.mode_backend import multi_level_kv_cache as impl
    from lightllm.server.multi_level_kv_cache import CacheTier

    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    m = MultiLevelKvCacheModule.__new__(MultiLevelKvCacheModule)
    m.backend = NS(is_master_in_dp=rank == 0, dp_world_size=2, is_hybrid_att_model=False)
    m.args = NS(cpu_cache_token_page_size=8, enable_disk_cache=False)
    m.async_cache = True
    m.incremental_offload = False
    m.gloo_group = dist.new_group(backend="gloo")
    m.filter_group = dist.new_group(backend="gloo")
    m.offload_sync_group = dist.new_group(backend="nccl")
    m.offload_sync_tensor = torch.zeros(1, device="cuda")
    m.load_tasks = deque()
    m.store_tasks = deque()
    m.checkpoint_tasks = deque()
    m.cache_sessions = {}
    owner = AttentionStateManager(1)
    state = GQAFullAttentionState(
        GQAAttentionConfig(
            logical_layer_groups=((0,),),
            global_head_num=2,
            head_dim=16,
            dtype=torch.bfloat16,
            tp_rank=rank,
            tp_world_size=2,
        ),
        page_size=4,
        device=f"cuda:{rank}",
    )
    state.allocate_storage(16)
    state.k_storage.fill_(42 + rank)
    state.v_storage.fill_(67 + rank)
    owner.register_state("main", state)
    owner.finish_state_registration()
    m.backend.model = NS(
        att_state_manager=owner, req_manager=NS(req_to_token_indexs=torch.arange(16, device="cuda").unsqueeze(0))
    )
    m.cpu_cache_operator = owner.get_cpu_cache_operator()
    size = m.cpu_cache_operator.get_page_bytes()
    m.cpu_cache_client = Client()
    m.cpu_cache_client.cpu_kv_cache_tensor = torch.full((64, size), 91, dtype=torch.uint8, pin_memory=True)
    m.staging_slots = [StagingSlot(torch.empty((4, size), dtype=torch.uint8, device="cuda")) for _ in range(2)]
    producer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    cpu = torch.cuda.Stream()
    impl.g_infer_context.get_overlap_stream = lambda: producer
    impl.g_infer_context.get_cpu_kv_cache_stream = lambda: cpu
    impl.g_infer_context.is_hybrid_att_model = False
    req = NS(
        req_id=123,
        req_idx=0,
        cur_kv_len=16,
        shared_kv_node=None,
        cache_tiers=(CacheTier.CPU,),
        shm_req=NS(
            group_req_id=123,
            request_id=123,
            input_len=17,
            token_hash_page_len_list=NS(get_all=lambda: [8, 16]),
            token_hash_list=NS(get_all=lambda: [10, 20]),
        ),
    )
    m.cache_sessions[123] = CacheSession(123)
    # Warm kernel compilation and first NCCL communicator before timing the delayed shard.
    stage = m.staging_slots[0].buffer[:2]
    m.cpu_cache_operator.write_req(
        torch.arange(16, device="cuda"), stage, [0, 1], m.cpu_cache_operator.get_page_layouts()
    )
    m.cpu_cache_operator.scatter_staging_pages(stage, m.cpu_cache_client.cpu_kv_cache_tensor, [0, 1])
    dist.all_reduce(m.offload_sync_tensor, group=m.offload_sync_group)
    torch.cuda.synchronize()
    m.cpu_cache_client.cpu_kv_cache_tensor.fill_(91)
    dist.barrier(group=m.gloo_group)
    if rank == 1:
        with torch.cuda.stream(cpu):
            torch.cuda._sleep(3_000_000_000)
    import time

    start = time.perf_counter()
    m.offload_finished_reqs_to_cpu_cache([req])
    print(rank, "submit_seconds", time.perf_counter() - start, "done", m.store_tasks[0].event.query(), flush=True)
    m._poll_async_tasks()
    if rank == 0:
        assert not m.cpu_cache_client.ready, "published before other TP shard completed"
    torch.cuda.synchronize()
    m._poll_async_tasks()
    assert not m.store_tasks and not m.cache_sessions
    direct = torch.full((2, size), 91, dtype=torch.uint8, pin_memory=True)
    m.cpu_cache_operator.write_req(
        torch.arange(16, device="cuda"), direct, [0, 1], m.cpu_cache_operator.get_page_layouts()
    )
    torch.cuda.synchronize()
    assert torch.equal(direct, m.cpu_cache_client.cpu_kv_cache_tensor[:2])
    if rank == 0:
        assert m.cpu_cache_client.ready == {0, 1} and m.cpu_cache_client.refs == {0: 0, 1: 0}
        print("TP2 NCCL completion/lifetime/scatter checks passed", flush=True)
    # A master pack event alone is not proof that the peer's pinned source was consumed.
    from lightllm.common.attention_state.allocators import StatePageAllocator
    from lightllm.server.router.model_infer.mode_backend.multi_level_kv_cache import CheckpointTask

    allocator = StatePageAllocator(1)
    ids = allocator.alloc(1)
    allocator.retain(ids)
    allocator.free(ids)
    with torch.cuda.stream(producer):
        if rank == 1:
            torch.cuda._sleep(1_000_000_000)
        packed = torch.cuda.Event()
        packed.record()
    if rank == 0:
        packed.synchronize()
    m.checkpoint_tasks.append(CheckpointTask(packed, [(allocator, ids)]))
    dist.barrier(group=m.gloo_group)
    m._poll_async_tasks()
    assert packed.query() and not m.checkpoint_tasks
    assert allocator.alloc(1) == ids
    # Capacity-pressure reclaim must also keep both ranks' slot IDs/order identical.
    ids = allocator.alloc(0)  # no-op contract
    occupied = [0]
    allocator.retain(occupied)
    allocator.free(occupied)
    with torch.cuda.stream(producer):
        if rank == 1:
            torch.cuda._sleep(100_000_000)
        pending = torch.cuda.Event()
        pending.record()
    m.checkpoint_tasks.append(CheckpointTask(pending, [(allocator, occupied)]))
    allocator.on_allocation_pressure = m._reclaim_checkpoint_slots
    assert allocator.alloc(1) == occupied and pending.query() and not m.checkpoint_tasks
    if rank == 0:
        print("TP2 checkpoint release and capacity-pressure checks passed", flush=True)
    torch.cuda.synchronize()
    dist.barrier(group=m.gloo_group)
    dist.destroy_process_group(m.offload_sync_group)
    dist.destroy_process_group(m.filter_group)
    dist.destroy_process_group(m.gloo_group)
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
def test_tp_completion_and_checkpoint_lifetimes():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(worker, args=(port,), nprocs=2, join=True)
