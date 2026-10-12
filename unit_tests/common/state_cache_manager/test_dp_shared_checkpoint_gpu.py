from datetime import timedelta
from types import SimpleNamespace as NS
import os
import uuid

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _config():
    return NS(
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
        linear_layer_num=2,
        get_conv_state_shape=lambda: (4, 3),
        get_ssm_state_shape=lambda: (2, 4, 3),
    )


def _gpu_rank(rank, rendezvous, service_name, kind):
    os.environ["LIGHTLLM_UNIQUE_SERVICE_NAME_ID"] = service_name
    os.environ["LIGHTLLM_CURRENT_DEVICE_ID"] = str(rank)
    # Follow production import order to avoid basemodel/req_manager circular initialization.
    from lightllm.server.router.model_infer.mode_backend import generic_pre_process  # noqa: F401
    from lightllm.common.state_cache_manager import LinearAttCacheManager, LayerCache
    from lightllm.common.kv_cache_mem_manager import qwen3next_mem_manager as memory_module
    from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextMemManager
    from lightllm.common.kv_cache_mem_manager.glm5_next_mem_manager import Glm5NextMemManager
    from lightllm.common.req_manager.linear_att import ReqManagerForMamba
    from lightllm.common.req_manager.glm5_next import Glm5NextReqManager

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    big = None
    try:
        config = _config()
        memory_type = Glm5NextMemManager if kind == "glm" else Qwen3NextMemManager
        memory = object.__new__(memory_type)
        memory_module.get_env_start_args = lambda: NS(
            dp=2, nnodes=1, linear_att_page_block_num=4, linear_att_hash_page_size=64
        )
        memory_module.create_new_group_for_current_node = lambda backend: dist.group.WORLD
        memory_module.get_dp_rank_in_node = lambda: rank
        memory_module.get_current_rank_in_dp = lambda: 0
        memory.size, memory.linear_config = 512, config
        memory._init_linear_att_buffers()
        big = memory.big_page_buffers
        small = LinearAttCacheManager(2, config)
        assert small.metadata_shm is None
        assert small.conv_state_cache.buffer.is_pinned()
        assert small.ssm_state_cache.buffer.is_pinned()
        small_slot = small.alloc_one_state_cache()
        for state in small.get_state_cache(small_slot):
            state.fill_(37 + rank)

        scratch = [memory.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID, memory.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID]
        scratch_ids = [None, None]
        dist.all_gather_object(scratch_ids, scratch)
        assert len(set(scratch_ids[0] + scratch_ids[1])) == 4
        assert all(index >= big.size - big.keep_num for index in scratch)
        for index in scratch:
            for state in big.get_state_cache(index):
                state.fill_(rank + 10)
        dist.barrier()
        for dp, indexes in enumerate(scratch_ids):
            for index in indexes:
                assert all(torch.all(state == dp + 10) for state in big.get_state_cache(index))

        manager_type = Glm5NextReqManager if kind == "glm" else ReqManagerForMamba
        req_manager = object.__new__(manager_type)
        req_manager.mem_manager, req_manager.small_page_buffers = memory, small
        req_manager.mtp_step = 0
        req_manager.req_to_mtp_state_index = None
        req_manager.req_to_conv_state = LayerCache(1, torch.bfloat16, (4, 3), 2, "cuda")
        req_manager.req_to_ssm_state = LayerCache(1, torch.float32, (2, 4, 3), 2, "cuda")
        if kind == "glm":
            req_manager.req_to_indexer_tail = NS(buffer=torch.ones((2, 1, 4, 8), device="cuda"))

        big_ids = [None]
        if rank == 1:
            slot = big.alloc_one_state_cache()
            big_ids[0] = slot
            req_manager.req_to_conv_state.buffer.fill_(19)
            req_manager.req_to_ssm_state.buffer.fill_(23)
            # Exercise the real fused Triton checkpoint save into registered shared storage.
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                req_manager.save_big_page_states(torch.tensor([0], device="cuda", dtype=torch.int32), [0], [slot])
            torch.cuda.synchronize()
            assert all(state.is_pinned() for state in big.get_state_cache(slot))
        # Test harness exchanges only an ID; no production DP-fetch flow is involved.
        dist.broadcast_object_list(big_ids, src=1)
        slot = big_ids[0]
        if rank == 0:
            big.retain_state_cache([slot])
        dist.barrier()
        assert big.refs[slot] == 2
        if rank == 1:
            # Releasing the source owner must not reclaim another process's checkpoint.
            big.free_state_cache([slot])
            another = big.alloc_one_state_cache()
            assert another != slot
            big.free_state_cache([another])
        dist.barrier()
        assert big.refs[slot] == 1
        if rank == 0:
            req = NS(req_idx=0)
            req_manager.restore_state(req, big, slot)
            torch.cuda.synchronize()
            assert torch.all(req_manager.req_to_conv_state.buffer == 19)
            assert torch.all(req_manager.req_to_ssm_state.buffer == 23)
            if kind == "glm":
                assert torch.all(req_manager.req_to_indexer_tail.buffer == 0)
            # Small checkpoints remain private and use the original restore interface.
            req_manager.restore_state(req, small, small_slot)
            torch.cuda.synchronize()
            assert torch.all(req_manager.req_to_conv_state.buffer == 37)
            assert torch.all(req_manager.req_to_ssm_state.buffer == 37)
            big.free_state_cache([slot])
        dist.barrier()
        assert big.get_free_cache_num() == 4
        assert all(torch.all(state == 37 + rank) for state in small.get_state_cache(small_slot))
        small.free_state_cache([small_slot])
        assert small.get_free_cache_num() == 2
        dist.barrier()
    finally:
        if rank == 0 and big is not None:
            for shm in (
                big.metadata_shm,
                big.lock.shm,
                big.conv_state_cache.shm,
                big.ssm_state_cache.shm,
            ):
                shm.unlink()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("kind", ["linear", "glm"])
def test_shared_checkpoint_save_restore_and_private_small_pool(tmp_path, kind):
    mp.spawn(
        _gpu_rank,
        args=(str(tmp_path / "rendezvous"), f"test_gpu_checkpoints_{uuid.uuid4().hex}", kind),
        nprocs=2,
        join=True,
    )
