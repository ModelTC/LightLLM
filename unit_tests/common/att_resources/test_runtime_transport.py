from __future__ import annotations

from unit_tests.common.att_resources.model_helpers import full_state_test_fields

from unit_tests.common.att_resources.model_helpers import full_state_test_buffers

from unit_tests.common.att_resources.model_helpers import read_kv_storage

from typing import TYPE_CHECKING
from unit_tests.common.att_resources.model_helpers import main_linear_state
from types import SimpleNamespace
import traceback
import uuid

import pytest
import torch
import torch.multiprocessing as mp

from lightllm.common.att_resources import AttResourceManager
from lightllm.common.att_resources.allocators import FullAttKVAllocator
from lightllm.server.router.model_infer import shared_att_resources_utils
from lightllm.server.router.model_infer.mode_backend.pd.resource_io import allocate_transfer_buffer, copy_page
from unit_tests.common.att_resources.test_model_resources import make_resources
from unit_tests.common.att_resources.test_quantized_full_state import KINDS, make_quant_state, assert_token_fields_equal

if TYPE_CHECKING:
    from easydict import EasyDict
    from multiprocessing.connection import Connection


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cpu_pd_and_dp_model_layouts_cover_quantized_fields(
    kind: str, device: str, pd_args: EasyDict, monkeypatch: pytest.MonkeyPatch,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pd_args.cpu_cache_token_page_size = pd_args.pd_kv_page_size = 4
    src, dst = make_quant_state(kind, device), make_quant_state(kind, device)
    managers = []
    for state in (src, dst):
        manager = AttResourceManager(request_capacity=2)
        manager.set_att_state("main", state)
        manager.full_att_allocator = FullAttKVAllocator(8, 4, pin_memory=False)
        manager.finish_state_creation()
        managers.append(manager)
    src, dst = [manager.get_full_attention_states("main")[0] for manager in managers]
    for i, field in enumerate(full_state_test_fields(src, full_state_test_buffers(src)).values()):
        if "token" in field.axes:
            field.tensor.fill_(i + 3)
    source_indices = torch.tensor([1, 3], dtype=torch.int32, device=device)
    pd_indices = torch.tensor([4, 6], dtype=torch.int64, device=device)
    cpu_indices = torch.tensor([5, 7], dtype=torch.int64, device=device)
    dp_indices = torch.tensor([0, 2], dtype=torch.int64, device=device)

    def reject_host_list(tensor: torch.Tensor) -> list[int]:
        pytest.fail("transfer paths must not convert Tensor indices to Python lists")

    monkeypatch.setattr(torch.Tensor, "tolist", reject_host_list)
    page = allocate_transfer_buffer(managers[0], 1, device=device)
    if device == "cpu":
        with pytest.raises(AssertionError, match="PD pages must reside on CUDA"):
            copy_page([managers[0]], page, page_index=0, dp_index=0, tp_size=1, token_indices=source_indices, write=True)
    else:
        copy_page([managers[0]], page, page_index=0, dp_index=0, tp_size=1, token_indices=source_indices, write=True)
        copy_page([managers[1]], page, page_index=0, dp_index=0, tp_size=1, token_indices=pd_indices, write=False)
        assert_token_fields_equal(src, [1, 3], dst, [4, 6])

    cpu_pages = torch.empty((1, managers[0].get_cpu_page_bytes()), dtype=torch.uint8, pin_memory=device == "cuda")
    if device == "cpu":
        with pytest.raises(AssertionError, match="CUDA State storage"):
            managers[0].write_req_to_cpu(source_indices, cpu_pages, [0], layouts=managers[0].get_cpu_transfer_layouts())
    else:
        managers[0].write_req_to_cpu(source_indices, cpu_pages, [0], layouts=managers[0].get_cpu_transfer_layouts())
        if device == "cuda":
            torch.cuda.synchronize()
        managers[1].read_req_from_cpu(cpu_pages, [0], cpu_indices, layouts=managers[1].get_cpu_transfer_layouts())
        assert_token_fields_equal(src, [1, 3], dst, [5, 7])

    if device == "cpu":
        with pytest.raises(AssertionError, match="DP token indices must reside on CUDA"):
            managers[1].copy_from_other_dp(
                remote_att_manager=managers[0], src_token_indexes=source_indices, dst_token_indexes=dp_indices,
                src_big_page_ids=[], dst_big_page_ids=[], src_req_idx=0, dst_req_idx=0,
            )
        return
    managers[1].copy_from_other_dp(
        remote_att_manager=managers[0],
        src_token_indexes=source_indices,
        dst_token_indexes=dp_indices,
        src_big_page_ids=[], dst_big_page_ids=[], src_req_idx=0, dst_req_idx=0,
    )
    assert_token_fields_equal(src, [1, 3], dst, [0, 2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("glm", [False, True])
@pytest.mark.parametrize("source_device", [0, 1])
@pytest.mark.parametrize("copy_big_pages", [False, True])
def test_dp_fetch_restores_baseline_runtime_and_full_token_fields(
    monkeypatch: pytest.MonkeyPatch,
    glm: bool,
    source_device: int,
    copy_big_pages: bool,
):
    if source_device >= torch.cuda.device_count():
        pytest.skip("two CUDA devices required")
    from lightllm.server.router.model_infer.mode_backend.dp_backend import dp_shared_kv_trans as module

    with torch.cuda.device(source_device):
        source, source_full = make_resources(glm, 1, 0, 2)
    with torch.cuda.device(0):
        target, target_full = make_resources(glm, 1, 0, 2)
    for i, field in enumerate(full_state_test_fields(source_full, full_state_test_buffers(source_full)).values()):
        if "token" in field.axes:
            field.tensor.fill_(i + 5)
    main_linear_state(source).conv_state.fill_(-1)
    main_linear_state(source).conv_state[:, 1, ..., :2].fill_(13)
    main_linear_state(source).ssm_state[:, 3].fill_(17)
    if glm:
        source.main_kpool_c4_bf16_state.tail_buffer[:, 1].fill_(23)
        target.main_kpool_c4_bf16_state.tail_buffer[:, 2].fill_(29)
    torch.cuda.synchronize(source_device)
    # 同时验证运行态与指定大页复制，保留其他页及小页快照。
    for pool in (main_linear_state(target).big_pages_pool, main_linear_state(target).small_pages_pool):
        pool.conv_state.fill_(41)
        pool.ssm_state.fill_(43)
    src_big_page_ids, dst_big_page_ids = [], []
    if copy_big_pages:
        src_big_page_ids = source.bounded_att_big_page_allocator.alloc(1)
        dst_big_page_ids = target.bounded_att_big_page_allocator.alloc(1)
        source_pool = main_linear_state(source).big_pages_pool
        source_pool.conv_state[src_big_page_ids[0]].fill_(51)
        source_pool.ssm_state[src_big_page_ids[0]].fill_(53)
    req = SimpleNamespace(req_idx=2, cur_kv_len=4, hold_kv_len=8, hybrid_len_to_big_page_id={})
    task = module.TransTask(
        dst_req=req, src_dp_rank=1, src_req_idx=1,
        src_token_indexes=torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=f"cuda:{source_device}"),
        dst_token_indexes=torch.tensor([8, 9, 10, 11]),
        src_big_page_ids=src_big_page_ids, dst_big_page_ids=dst_big_page_ids,
    )
    syncs = []

    def barrier(group: str):
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(source_device)
        assert req.cur_kv_len == 4
        assert not req.hybrid_len_to_big_page_id
        syncs.append(group)

    monkeypatch.setattr(module.dist, "barrier", barrier)
    worker = module.DPKVSharedMoudle.__new__(module.DPKVSharedMoudle)
    worker.backend = SimpleNamespace(
        model=SimpleNamespace(att_resource_manager=target),
        dp_att_resource_managers={1: source},
        is_hybrid_att_model=True,
        is_master_in_dp=False,
        node_nccl_group="node",
    )
    worker.big_page_tokens = 4
    worker.kv_trans([task])
    assert syncs == ["node"] and req.cur_kv_len == 8
    assert_token_fields_equal(source_full, [0, 1, 2, 3], target_full, [8, 9, 10, 11])
    assert (main_linear_state(target).conv_state[:, 2, ..., :2] == 13).all()
    assert (main_linear_state(target).ssm_state[:, 6] == 17).all()
    assert main_linear_state(target).mtp_state_index[2] == 0
    assert not main_linear_state(target).conv_state[:, :2].any()
    big_pool = main_linear_state(target).big_pages_pool
    if copy_big_pages:
        page_id = dst_big_page_ids[0]
        assert (big_pool.conv_state[page_id] == 51).all()
        assert (big_pool.ssm_state[page_id] == 53).all()
        assert req.hybrid_len_to_big_page_id == {8: page_id}
    else:
        assert not req.hybrid_len_to_big_page_id
    untouched = [page for page in range(big_pool.size) if page not in dst_big_page_ids]
    assert (big_pool.conv_state[untouched] == 41).all()
    assert (big_pool.ssm_state[untouched] == 43).all()
    assert (main_linear_state(target).small_pages_pool.conv_state == 41).all()
    assert (main_linear_state(target).small_pages_pool.ssm_state == 43).all()
    if glm:
        assert not target.main_kpool_c4_bf16_state.tail_buffer[:, 2].any()
        assert (source.main_kpool_c4_bf16_state.tail_buffer[:, 1] == 23).all()
        assert not target.main_kpool_c4_bf16_state.tail_buffer[:, :2].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("p2p", [False, True])
@pytest.mark.parametrize("share_pools,ipc_kinds", [
    (False, ("pd",)),
    (True, ("pd", "dp")),
    (True, ("dp", "pd")),
])
def test_actual_rank_storage_ipc_separates_pd_and_dp_exports(
    monkeypatch: pytest.MonkeyPatch, p2p: bool, share_pools: bool,
    ipc_kinds: tuple[str, ...], cpu_args: EasyDict,
):
    cpu_args.enable_dp_prompt_cache_fetch = share_pools
    from lightllm.utils.envs_utils import get_unique_server_name
    from lightllm.utils.shm_utils import create_or_link_shm

    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", "att-ipc-" + uuid.uuid4().hex)
    get_unique_server_name.cache_clear()
    monkeypatch.setattr(shared_att_resources_utils, "get_current_rank_in_node", lambda: 0)
    if p2p and torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices required for IPC peer mapping")
    monkeypatch.setattr(shared_att_resources_utils, "kv_trans_use_p2p", lambda: p2p)
    monkeypatch.setattr(mp.reductions.reduce_tensor, "__code__", mp.reductions.reduce_tensor.__code__)
    source_device = int(p2p)
    with torch.cuda.device(source_device):
        resources, full = make_resources(True, 1, 0, 2)
    for storage_buffer in full_state_test_buffers(full).values():
        storage_buffer.fill_(9)
    for storage_buffer in full_state_test_buffers(full).values():
        storage_buffer[:, 4].zero_()
    main_linear_state(resources).conv_state.fill_(11)
    resources.main_kpool_c4_bf16_state.tail_buffer.fill_(7)
    for state in resources.get_bounded_attention_states():
        for pool in (state.big_pages_pool, state.small_pages_pool):
            if pool is not None:
                pool.conv_state.zero_()
                pool.ssm_state.zero_()
    slot = resources.bounded_att_big_page_allocator.alloc(1)
    resources.save_bounded_att_pages([0], page_kind="big", slot_ids=slot)
    mapping = torch.zeros((3, 16), dtype=torch.int32, device=f"cuda:{source_device}")
    mapping[1, 0] = 7
    torch.cuda.synchronize(source_device)
    for kind in ipc_kinds:
        shared_att_resources_utils.publish_att_resources(resources, mapping, kind=kind, reader_count=1)
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_ipc_worker, args=(child, ipc_kinds))
    process.start()
    child.close()
    try:
        assert parent.poll(45), "CUDA IPC child did not finish"
        error = parent.recv()
        assert error is None, error
        assert (read_kv_storage(full)[:, 4] == 9).all()
        assert not resources.main_kpool_c4_bf16_state.tail_buffer[:, 1].any()
        assert (main_linear_state(resources).conv_state == 11).all()
        assert (main_linear_state(resources).big_pages_pool.conv_state[slot[0]] == 11).all()
        assert not main_linear_state(resources).big_pages_pool.ssm_state[slot[0]].any()
        if share_pools:
            assert main_linear_state(resources).small_pages_pool.conv_state.flatten()[0].item() == 37
        assert resources.bounded_att_big_page_allocator.free_count == 1
        parent.send(None)
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
        for kind in ipc_kinds:
            shm = create_or_link_shm(f"rank_storage_{kind}_0", expected_size=-1, force_mode="link")
            shm.close()
            shm.unlink()
        get_unique_server_name.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("source_tp,target_tp", [(1, 4), (4, 1), (2, 8), (8, 2)])
def test_qwen_pd_reshards_full_and_linear_replicated_heads(source_tp: int, target_tp: int, pd_args: EasyDict):
    pd_args.pd_kv_page_size = 128
    source = [make_resources(False, source_tp, rank, 2) for rank in range(source_tp)]
    target = [make_resources(False, target_tp, rank, 2) for rank in range(target_tp)]
    for rank, (manager, full) in enumerate(source):
        for field_index, field in enumerate(full_state_test_fields(full, full_state_test_buffers(full)).values()):
            for local_head in range(full.config.head_num):
                head = rank * full.config.head_num // max(1, source_tp // 2) + local_head
                field.tensor[:, :, local_head].fill_(100 * field_index + head)
        cfg = main_linear_state(manager).config
        offset = 0
        for group, (local_heads, global_heads) in enumerate(
            ((cfg.num_k_heads, 2), (cfg.num_k_heads, 2), (cfg.num_v_heads, 4))
        ):
            start = rank * local_heads // max(1, source_tp // global_heads)
            for head in range(local_heads):
                main_linear_state(manager).conv_state[:, 1, offset + head * 8 : offset + (head + 1) * 8, :2].fill_(
                    10 * group + start + head
                )
            offset += local_heads * 8
        start = rank * cfg.num_v_heads // max(1, source_tp // 4)
        for head in range(cfg.num_v_heads):
            main_linear_state(manager).ssm_state[:, 3, head].fill_(30 + start + head)
    page = allocate_transfer_buffer(source[0][0], 1, device="cuda")
    for kind in ("kv", "att_state"):
        copy_page(
            [m for m, _ in source],
            page,
            page_index=0,
            dp_index=0,
            tp_size=source_tp,
            token_indices=torch.tensor([0, 3], dtype=torch.long, device="cuda"),
            kind=kind,
            req_idx=1,
            write=True,
        )
        copy_page(
            [m for m, _ in target],
            page,
            page_index=0,
            dp_index=0,
            tp_size=target_tp,
            token_indices=torch.tensor([4, 7], dtype=torch.long, device="cuda"),
            kind=kind,
            req_idx=2,
            write=False,
        )
    for rank, (manager, full) in enumerate(target):
        for field_index, field in enumerate(full_state_test_fields(full, full_state_test_buffers(full)).values()):
            for local_head in range(full.config.head_num):
                head = rank * full.config.head_num // max(1, target_tp // 2) + local_head
                assert (field.tensor[:, [4, 7], local_head] == 100 * field_index + head).all()
        cfg = main_linear_state(manager).config
        offset = 0
        for group, (local_heads, global_heads) in enumerate(
            ((cfg.num_k_heads, 2), (cfg.num_k_heads, 2), (cfg.num_v_heads, 4))
        ):
            start = rank * local_heads // max(1, target_tp // global_heads)
            for head in range(local_heads):
                actual = main_linear_state(manager).conv_state[:, 2, offset + head * 8 : offset + (head + 1) * 8, :2]
                assert (actual == 10 * group + start + head).all()
            offset += local_heads * 8
        start = rank * cfg.num_v_heads // max(1, target_tp // 4)
        for head in range(cfg.num_v_heads):
            assert (main_linear_state(manager).ssm_state[:, 6, head] == 30 + start + head).all()
        assert main_linear_state(manager).mtp_state_index[2] == 0


def _ipc_worker(connection: Connection, ipc_kinds: tuple[str, ...]) -> None:
    try:
        torch.cuda.set_device(0)
        for kind in ipc_kinds:
            resources, mapping = shared_att_resources_utils.load_att_resources(0, kind=kind)
            assert resources._states_created
            assert resources.full_att_allocator is None
            assert resources.bounded_att_big_page_allocator is None
            assert resources.bounded_att_small_page_allocator is None
            if kind == "dp":
                for state in resources.get_bounded_attention_states():
                    assert state.big_pages_pool.conv_state.is_shared()
                    assert not state.big_pages_pool.conv_state.is_pinned()
                resources.register_page_pools_pin_memory()
                resources.register_page_pools_pin_memory()
            for state in resources.get_bounded_attention_states():
                for pool in (state.big_pages_pool, state.small_pages_pool):
                    if kind == "dp":
                        assert pool is not None
                        assert pool.conv_state.is_pinned() and pool.ssm_state.is_pinned()
                        copied = pool.conv_state.to(device="cuda", non_blocking=True)
                        torch.cuda.synchronize()
                        assert torch.equal(copied.cpu(), pool.conv_state)
                    else:
                        assert pool is None
            if kind == "dp":
                main_linear_state(resources).small_pages_pool.conv_state.flatten()[0] = 37
            assert not hasattr(resources, "req_to_token_indexs")
            assert mapping[1, 0].item() == 7
            resources.copy_full_att_tokens(torch.tensor([0], dtype=torch.long, device="cuda"), torch.tensor([4], dtype=torch.long, device="cuda"))
            resources.reset_attention_tail_state([1])
            torch.cuda.synchronize()
            # 一类通道耗尽不会影响另一类通道的读取计数。
            with pytest.raises(RuntimeError, match="exhausted"):
                shared_att_resources_utils.load_att_resources(0, kind=kind)
        connection.send(None)
        # Keep imported tensors alive until the parent has checked the writes.
        connection.recv()
    except BaseException:
        connection.send(traceback.format_exc())
    finally:
        AttResourceManager._get_states.cache_clear()
        connection.close()
