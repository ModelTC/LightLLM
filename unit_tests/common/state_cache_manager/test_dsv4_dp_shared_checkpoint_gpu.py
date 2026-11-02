"""Node-local cross-DP KV fetch with real CUDA IPC and shared checkpoints."""

import gc
import json
import os
import socket
import uuid
from datetime import timedelta
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _rank(rank, rendezvous, service_name, model_dir, model_type, end, start, mtp_step):
    for name in ("GLOBAL_RANK", "CURRENT_DEVICE_ID", "CURRENT_RANK_IN_NODE", "GLOBAL_DP_RANK", "DP_RANK_IN_NODE"):
        os.environ["LIGHTLLM_" + name] = str(rank)
    os.environ["LIGHTLLM_CURRENT_RANK_IN_DP"] = "0"
    os.environ["LIGHTLLM_GLOBAL_WORLD_SIZE"] = os.environ["LIGHTLLM_NODE_WORLD_SIZE"] = "2"
    os.environ["LIGHTLLM_DP_WORLD_SIZE"] = "1"
    os.environ["LIGHTLLM_DP_SIZE"] = "2"
    os.environ["LIGHTLLM_UNIQUE_SERVICE_NAME_ID"] = service_name
    torch.cuda.set_device(rank)
    import lightllm.common.basemodel  # noqa: F401
    from lightllm.common.kv_cache_mem_manager import DeepseekV4MemoryManager, MemoryManager
    from lightllm.common.req_manager import DeepseekV4ReqManager
    from lightllm.common.state_cache_manager.deepseek4 import DeepseekV4CacheConfig
    from lightllm.models.deepseek_v4.triton_kernel.swa_cache import gather_swa_state
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.server.router.model_infer.infer_batch import InferReq
    from lightllm.server.router.model_infer.mode_backend.dp_backend import dp_shared_kv_trans as transfer
    from lightllm.utils.envs_utils import set_env_start_args
    from lightllm.utils.service_shm_cleanup import ServiceShmCleanup

    args = StartArgs(
        model_dir=model_dir,
        tp=2,
        dp=2,
        nnodes=1,
        page_size=256,
        max_req_total_len=1024,
        linear_att_hash_page_size=256,
        linear_att_page_block_num=2,
        run_mode="normal",
        running_max_req_size=2,
        enable_dp_prompt_cache_fetch=True,
    )
    set_env_start_args(args)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        nccl_group = dist.new_group([0, 1], backend="nccl")
        if model_type == "deepseek_v41":
            config = DeepseekV4CacheConfig(
                (0, 0) + (2,) * 18 + (1,) * 20 + (0,) * (3 if mtp_step else 0),
                512,
                64,
                128,
                128,
                kv_source_layer_ids=(2, 8, 14, 20),
                indexer_layer_ids=(2, 8, 14, 20),
                short_compress_ratio=2,
                long_compress_ratio=1,
            )
        else:
            config = DeepseekV4CacheConfig((4, 128), 512, 64, 128, 128)
        memory = DeepseekV4MemoryManager(1536, torch.uint8, 1, 512, config.layer_num, config)
        requests = DeepseekV4ReqManager(2, 1024, memory, config, mtp_step=mtp_step)
        small = requests.create_small_page_cache_manager(4)
        big = memory.big_page_buffers
        source_indexes = torch.cat([torch.arange(a, a + 256, device="cuda") for a in (512, 0, 256)]).int()
        req = InferReq.__new__(InferReq)
        req.args, req.req_id, req.req_idx = args, 7, 0
        req.cur_kv_len, req.hold_kv_len = 0, 0
        req.cur_output_len = 0
        req.shm_req = NS(input_len=end + 1, shm_cur_kv_len=0, prompt_cache_len=0)
        req.hybrid_cache_len, req.hybrid_len_to_big_page_id = end, {}
        req.tail_small_page_buffer_id = None

        def pattern(index, width):
            return ((torch.arange(end * width, device="cuda").view(end, 1, width) + index * 17) % 251).byte()

        owned = []
        ids = []
        if rank == 1:
            requests.req_to_token_indexs[0, :end] = source_indexes[:end]
            for index, buffer in enumerate(memory.kv_buffer):
                buffer.index_copy_(0, source_indexes[:end].long(), pattern(index, buffer.shape[-1]))
            for length, pool, value in ((512, big, 19), (768, small, 23)):
                if length > end:
                    continue
                slot = pool.alloc_one_state_cache()
                requests.req_to_swa[:, 0].fill_(value)
                requests.req_to_c4_state[:, 0].fill_(value + 1)
                requests.req_to_c4_indexer_state[:, 0].fill_(value + 2)
                req.get_chuncked_input_token_len = lambda length=length: length
                requests.save_state(req, slot, pool)
                ids.append(slot)
                owned.append((pool, slot))
            # A receiver adopts existing IDs even when checkpoint pools are full.
            other_slots = [(pool, pool.alloc_state_cache(pool.get_free_cache_num())) for pool in (big, small)]
        else:
            req.cur_kv_len = req.hold_kv_len = start
            if start:
                local = memory.alloc(start)
                requests.req_to_token_indexs[0, :start] = local.cuda()
                for index, buffer in enumerate(memory.kv_buffer):
                    buffer[local.long().cuda()] = pattern(index, buffer.shape[-1])[:start]

        torch.cuda.synchronize()
        dist.barrier()
        memory.write_to_shm(requests)
        dist.barrier()
        memories = [memory if peer == rank else MemoryManager.loads_from_shm(peer) for peer in range(2)]
        cache = NS(
            get_mem_index_value_by_node=lambda node, begin, finish: source_indexes[begin:finish].cpu(),
            get_big_page_ids_by_node=lambda node: [ids[0]],
        )
        backend = NS(
            args=args,
            dp_size_in_node=2,
            dp_world_size=1,
            rank_in_dp=0,
            is_master_in_dp=True,
            node_gloo_group=dist.group.WORLD,
            node_nccl_group=nccl_group,
            model=NS(mem_manager=memory, req_manager=requests),
            small_page_buffers=small,
            radix_cache=cache,
            logger=NS(info=lambda _: None),
        )

        def allocate(request, count):
            slots = memory.alloc(count)
            requests.req_to_token_indexs[0, request.hold_kv_len : request.hold_kv_len + count] = slots.cuda()
            request.hold_kv_len += count
            return slots

        backend._alloc_req_kv_mem = allocate
        module = object.__new__(transfer.DPKVSharedMoudle)
        module.backend, module.dp_rank_in_node = backend, rank
        module.init_dsv4_cache_transfer(memories)
        transfer.g_infer_context = NS(req_manager=requests, get_can_alloc_token_num=lambda: 1536 - start)
        node = NS(small_page_buffer_idx=ids[-1]) if rank == 1 else None
        tasks, plan = module.build_dsv4_trans_tasks(
            [(7, 0, None, 0)], [req] if rank == 0 else [], np.array([[start, end]])
        )
        assert len(tasks) == (1 if rank == 0 else 0)
        assert plan == [[(1, 7, start, end)], []]
        module.kv_trans_dsv4(tasks, {7: NS(node=node, matched_len=end)}, plan)
        dist.barrier()
        if rank == 1:
            for pool, slot in owned:
                assert pool.refs[slot] == 2
                pool.free_state_cache([slot])
                assert pool.refs[slot] == 1
                assert pool.alloc_one_state_cache() is None
            for buffer in memory.kv_buffer:
                buffer.fill_(255)
        dist.barrier()
        if rank == 0:
            assert req.cur_kv_len == req.hold_kv_len == req.shm_req.shm_cur_kv_len == end
            destination = requests.req_to_token_indexs[0, :end].long()
            for index, buffer in enumerate(memory.kv_buffer):
                torch.testing.assert_close(buffer[destination], pattern(index, buffer.shape[-1]), atol=0, rtol=0)
            pool = big if end == 512 else small
            slot = req.hybrid_len_to_big_page_id[end] if end == 512 else req.tail_small_page_buffer_id
            requests.req_to_swa.zero_()
            requests.restore_state(req, pool, slot)
            window = gather_swa_state(requests, 0, torch.arange(end - 128, end, device="cuda"))
            value = 19 if end == 512 else 23
            assert torch.all(window == value)
            if config.short_compress_ratio == 4:
                positions = torch.arange(end - 4, end, device="cuda") % requests.c4_state_tokens
                assert torch.all(requests.req_to_c4_state[:, 0, positions] == value + 1)
                assert torch.all(requests.req_to_c4_indexer_state[:, 0, positions] == value + 2)
            if model_type == "deepseek_v41":
                from lightllm.models.deepseek_v41 import engram
                from lightllm.server.router.model_infer import infer_batch
                from lightllm.server.router.model_infer.mode_backend import generic_pre_process

                # A remote KV/checkpoint hit carries no process-local Engram history.
                # Restore the CPU lookback at the actual encoder replay start.
                allocate(req, 256)
                req.token_history_ready = False
                req.mm_input_spans = []
                req.multimodal_params = {"images": [], "audios": []}
                tokens = np.arange(end + 1, dtype=np.int64) % 64
                req.shm_req.shm_prompt_ids = NS(arr=tokens)
                engram.build_compressed_token_map = lambda _: (list(range(64)), 64)
                hasher = engram.EngramHasher(
                    dict(
                        engram_compressed_vocab_size=64,
                        engram_pad_token_id=2,
                        engram_max_ngram_size=4,
                        engram_layer_ids=[1, 14],
                        engram_vocab_size=101,
                        engram_n_heads=2,
                    ),
                    None,
                    memory.size + memory.page_size,
                )

                def hash_rows(begin):
                    positions = torch.arange(begin, end + 1, device="cuda")
                    return hasher(
                        NS(
                            position_ids=positions,
                            input_ids=torch.as_tensor(tokens[begin:], device="cuda"),
                            dsv4_sparse_req_idx=torch.zeros_like(positions),
                            mem_index=requests.req_to_token_indexs[0, positions],
                            req_manager=requests,
                        )
                    )

                expected = hash_rows(0)[-128:].clone()
                hasher.history.fill_(-99)
                context = NS(backend=NS(model=NS(token_lookback_depth=3, prefill_replay_window=128)))
                infer_batch.g_infer_context = generic_pre_process.g_infer_context = context
                model_input, _ = generic_pre_process.prepare_prefill_inputs([req], False)
                assert model_input.input_ids.numel() == 129
                assert model_input.b_ready_cache_len.tolist() == [end - 128]
                assert model_input.b_prefill_commit_start.tolist() == [end]
                hasher.restore_history(model_input.token_lookback, requests)
                torch.testing.assert_close(hash_rows(end - 128)[-128:], expected, atol=0, rtol=0)
                assert req.cur_kv_len == end
            for slot in req.hybrid_len_to_big_page_id.values():
                big.free_state_cache([slot])
            if req.tail_small_page_buffer_id is not None:
                small.free_state_cache([req.tail_small_page_buffer_id])
        dist.barrier()
        if rank == 1:
            for pool, slots in other_slots:
                pool.free_state_cache(slots)
        dist.barrier()
        assert big.get_free_cache_num() == big.size - big.keep_num
        assert small.get_free_cache_num() == small.size - small.keep_num
        del memories, module
        gc.collect()
        dist.barrier()
    finally:
        dist.destroy_process_group()
        if rank == 0:
            ServiceShmCleanup(service_name).cleanup_service_resources()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize(
    "model_type,end,start,mtp_step",
    [
        ("deepseek_v4", 768, 256, 0),
        ("deepseek_v41", 512, 0, 0),
        ("deepseek_v41", 768, 256, 5),
    ],
)
def test_fetch_copies_kv_and_restores_shared_checkpoints_after_source_eviction(
    tmp_path, model_type, end, start, mtp_step
):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type, "vocab_size": 64}))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        rendezvous = f"tcp://127.0.0.1:{sock.getsockname()[1]}"
    mp.spawn(
        _rank,
        args=(rendezvous, "test_dp_fetch_" + uuid.uuid4().hex, str(tmp_path), model_type, end, start, mtp_step),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_shared_checkpoint_copy_across_default_registration_boundary(monkeypatch):
    import lightllm.common.basemodel  # noqa: F401
    from lightllm.common.state_cache_manager.deepseek4 import DeepseekV4CacheConfig, DeepseekV4StateCacheManager
    from lightllm.utils.service_shm_cleanup import ServiceShmCleanup

    service_name = "test_v41_registration_" + uuid.uuid4().hex
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", service_name)
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    torch.cuda.set_device(0)
    config = DeepseekV4CacheConfig((0, 0, 2, 1), 512, 64, 128, 128, short_compress_ratio=2, long_compress_ratio=1)
    state_bytes = config.get_checkpoint_bytes()
    crossing = 128 * 1024 * 1024 // state_bytes
    assert crossing * state_bytes < 128 * 1024 * 1024 < (crossing + 1) * state_bytes
    try:
        pool = DeepseekV4StateCacheManager(crossing + 2, cache_config=config, shm_name="checkpoints", shm_create=True)
        source = torch.full((state_bytes,), 173, dtype=torch.uint8, device="cuda")
        pool.buffer[crossing].copy_(source, non_blocking=True)
        restored = pool.buffer[crossing].cuda(non_blocking=True)
        torch.cuda.synchronize()
        torch.testing.assert_close(restored, source, atol=0, rtol=0)
    finally:
        ServiceShmCleanup(service_name).cleanup_service_resources()
