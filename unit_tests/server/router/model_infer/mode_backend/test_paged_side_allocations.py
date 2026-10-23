from types import SimpleNamespace

import numpy as np
import pytest
import torch

# 先加载通用预处理模块，沿用生产初始化顺序，避免 basemodel 与 req_manager
# 在独立运行本测试文件时形成模块级循环导入。
from lightllm.server.router.model_infer.mode_backend import generic_pre_process  # noqa: F401
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.infer_req_ext1 import PinMemPageCacheReqTokenIndex, create_req_token_index
from lightllm.server.router.model_infer.mode_backend import base_backend
from lightllm.server.router.model_infer.mode_backend.dp_backend import dp_shared_kv_trans
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import decode_impl
from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl import prefill_impl
from unit_tests.common.cpu_index_helpers import make_cpu_index_manager, bind_cpu_indexes


def _bind_alloc_need(req, page_size):
    req.args = SimpleNamespace(page_size=page_size)
    req._kv_cache_alloc_need = lambda target_len: InferReq._kv_cache_alloc_need(req, target_len)
    return req


def _reserve_into_table(table, req, alloc_sizes, next_index, alloc_token_num):
    if alloc_token_num == 0:
        return None
    alloc_sizes.append(alloc_token_num)
    new_hold_kv_len = req.hold_kv_len + alloc_token_num
    mem_indexes = torch.arange(next_index[0], next_index[0] + alloc_token_num, dtype=torch.int32)
    table[req.req_idx, req.hold_kv_len : new_hold_kv_len] = mem_indexes
    next_index[0] += alloc_token_num
    req.hold_kv_len = new_hold_kv_len
    return mem_indexes


@pytest.mark.parametrize("is_hybrid_att_model", [False, True])
def test_pd_decode_reserves_model_pages_but_transfers_only_logical_kv(monkeypatch, is_hybrid_att_model):
    table = torch.full((1, 16), -1, dtype=torch.int32)
    table[0, :4] = torch.arange(4, dtype=torch.int32)
    alloc_sizes = []
    next_index = [4]
    tasks = []
    backend = decode_impl.PDDecodeNode.__new__(decode_impl.PDDecodeNode)
    backend.args = SimpleNamespace(pd_kv_page_size=4, page_size=4)
    backend.model = SimpleNamespace(req_manager=make_cpu_index_manager(token_table=table, page_size=1))
    backend.is_master_in_dp = False

    def create_task(**kwargs):
        tasks.append(
            (
                kwargs["kv_start_index"],
                kwargs["kv_end_index"],
                kwargs["mem_indexes"],
                kwargs.get("page_kind", "kv"),
            )
        )
        kwargs["group"].task_list.append(SimpleNamespace())

    backend._create_pd_trans_task = create_task
    monkeypatch.setattr(decode_impl.g_infer_context, "is_hybrid_att_model", is_hybrid_att_model)
    req = _bind_alloc_need(
        SimpleNamespace(
            req_idx=0,
            cur_kv_len=4,
            hold_kv_len=4,
            pd_trans_kv_start_index=0,
            shm_req=SimpleNamespace(input_len=10),
        ),
        page_size=4,
    )

    req.alloc_kv_mem = lambda size: _reserve_into_table(table, req, alloc_sizes, next_index, size)

    backend._decode_node_gen_trans_tasks(req)

    assert alloc_sizes == [8]
    assert req.cur_kv_len == 10
    assert req.hold_kv_len == 12
    expected_tasks = [
        (4, 8, [4, 5, 6, 7], "kv"),
        (8, 10, [8, 9], "kv"),
    ]
    if is_hybrid_att_model:
        expected_tasks.append((10, 10, [], "att_state"))
    assert tasks == expected_tasks
    assert table[0, :12].tolist() == list(range(12))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and pinned CPU memory")
@pytest.mark.parametrize("mode", ["pin_mem_cache", "pin_mem_page_cache", "pure_gpu"])
@pytest.mark.parametrize("input_len", [128, 130])
def test_pd_prefill_transfers_exact_logical_tail_with_real_index_helper(monkeypatch, mode, input_len):
    monkeypatch.setattr(PinMemPageCacheReqTokenIndex, "_page_offsets", None)
    monkeypatch.setattr(prefill_impl.g_infer_context, "is_hybrid_att_model", False)
    page_size = 64
    token_num = (input_len + page_size - 1) // page_size * page_size
    req = SimpleNamespace(
        req_idx=0,
        req_id=1,
        req_manager=SimpleNamespace(
            page_size=page_size, req_to_token_indexs=torch.empty((1, token_num), dtype=torch.int32, device="cuda")
        ),
        args=SimpleNamespace(req_token_index_mode=mode),
        shm_req=SimpleNamespace(input_len=input_len),
        sampling_param=SimpleNamespace(
            pd_master_node_id=1,
            pd_decode_node=SimpleNamespace(agent_name="decode", agent_metadata=None, num_pages=1, page_reg_desc=None),
        ),
        cur_kv_len=input_len,
        hold_kv_len=token_num,
        pd_trans_kv_start_index=0,
        pd_trans_device_id=0,
        pd_task_num=0,
    )
    req.token_index = create_req_token_index(req, index_stream=torch.cuda.default_stream())
    page_starts = torch.tensor([9, 2, 15], dtype=torch.int32)[: token_num // page_size] * page_size
    expected = (page_starts[:, None] + torch.arange(page_size, dtype=torch.int32)).reshape(-1)
    req.token_index.append_token_indexes(expected)
    backend = prefill_impl.PDChunkedPrefillForPrefillNode.__new__(prefill_impl.PDChunkedPrefillForPrefillNode)
    backend.args = SimpleNamespace(page_size=page_size, pd_kv_page_size=128)
    backend.is_master_in_dp = True
    backend.dp_rank_in_node = 0
    tasks = []
    backend.info_queue = SimpleNamespace(put=tasks.append)

    backend._prefill_chuncked_handle_func(req, next_token_id=7, next_token_prob=-0.5, output_len=1)

    assert req.pd_task_num == len(tasks)
    assert req.pd_trans_kv_start_index == input_len
    assert [(task.start_kv_index, task.end_kv_index) for task in tasks] == (
        [(0, 128)] if input_len == 128 else [(0, 128), (128, 130)]
    )
    assert [index for task in tasks for index in task.mem_indexes] == expected[:input_len].tolist()
    assert tasks[-1].first_gen_token_id == 7
    assert tasks[-1].first_gen_token_logprob == -0.5


def test_pd_decode_rejects_unaligned_transfer_page_size():
    backend = decode_impl.PDDecodeNode.__new__(decode_impl.PDDecodeNode)
    backend.args = SimpleNamespace(pd_kv_page_size=3, page_size=4)
    req = SimpleNamespace(cur_kv_len=4, shm_req=SimpleNamespace(input_len=10))

    with pytest.raises(AssertionError, match="pd_kv_page_size must be divisible by page_size"):
        backend._decode_node_gen_trans_tasks(req)


def test_pd_prefill_rejects_unaligned_transfer_page_size():
    backend = prefill_impl.PDChunkedPrefillForPrefillNode.__new__(prefill_impl.PDChunkedPrefillForPrefillNode)
    backend.args = SimpleNamespace(pd_kv_page_size=3, page_size=4)
    req = SimpleNamespace(cur_kv_len=4, shm_req=SimpleNamespace(input_len=10))

    with pytest.raises(AssertionError, match="pd_kv_page_size must be divisible by page_size"):
        backend._prefill_chuncked_handle_func(req, next_token_id=1, next_token_prob=1.0, output_len=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("is_master_in_dp", [False, True])
def test_dp_cache_fetch_publishes_gpu_indexes_before_barrier_returns(monkeypatch, is_master_in_dp):
    cpu_table = torch.empty((2, 16), dtype=torch.int32, pin_memory=True)
    torch.arange(32, out=cpu_table.view(-1))
    gpu_table = torch.full((2, 16), -1, dtype=torch.int32, device="cuda")
    gpu_table[1, :2].copy_(cpu_table[1, :2], non_blocking=True)
    req_manager = make_cpu_index_manager(token_table=cpu_table, req_to_token_indexs=gpu_table, page_size=4)
    reqs = [
        SimpleNamespace(req_idx=1, cur_kv_len=4, hold_kv_len=4),
        SimpleNamespace(req_idx=0, cur_kv_len=4, hold_kv_len=8),
    ]
    for req, gpu_index_submitted_len in zip(reqs, (2, 0)):
        bind_cpu_indexes(req, req_manager, gpu_index_submitted_len=gpu_index_submitted_len)
    default_stream = torch.cuda.default_stream()
    # 索引旧写入统一在默认流；上传流必须等待它，随后默认流再等待上传完成。
    torch.cuda._sleep(20_000_000)
    gpu_table[0, :8].fill_(-2)
    context = SimpleNamespace(req_manager=req_manager)
    monkeypatch.setattr(dp_shared_kv_trans, "g_infer_context", context)
    monkeypatch.setattr(base_backend, "g_infer_context", context)
    backend = SimpleNamespace(node_nccl_group=object(), is_master_in_dp=is_master_in_dp)
    order = []

    def upload(run_reqs):
        assert torch.cuda.current_stream() == default_stream
        order.append("upload")
        base_backend.ModeBackend._copy_req_token_indexes_to_gpu(backend, run_reqs)

    original_synchronize = torch.cuda.Stream.synchronize
    original_wait_stream = torch.cuda.Stream.wait_stream

    def forbid_synchronize(*args):
        pytest.fail("DP shared must rely on NCCL barrier rather than explicitly synchronizing a stream")

    def wait_stream(waiting_stream, source_stream):
        if waiting_stream != default_stream:
            assert source_stream == default_stream
            order.append("upload_waits_default")
        else:
            assert source_stream != default_stream
            order.append("default_waits_upload")
        original_wait_stream(waiting_stream, source_stream)

    def barrier(group):
        assert group is backend.node_nccl_group
        assert order == ["upload", "upload_waits_default", "default_waits_upload"]
        # 模拟同步 NCCL barrier 在 C++ 层等待默认流完成。
        original_synchronize(default_stream)
        assert torch.equal(gpu_table[1, :4].cpu(), cpu_table[1, :4])
        assert torch.equal(gpu_table[0, :8].cpu(), cpu_table[0, :8])
        assert gpu_table[0, 8:].eq(-1).all()
        assert gpu_table[1, 4:].eq(-1).all()
        order.append("barrier")

    backend._copy_req_token_indexes_to_gpu = upload
    monkeypatch.setattr(torch.cuda.Stream, "synchronize", forbid_synchronize)
    monkeypatch.setattr(torch.cuda.Stream, "wait_stream", wait_stream)
    monkeypatch.setattr(dp_shared_kv_trans.dist, "barrier", barrier)
    module = dp_shared_kv_trans.DPKVSharedMoudle.__new__(dp_shared_kv_trans.DPKVSharedMoudle)
    module.backend = backend
    module.dp_rank_in_node = 0
    module.shared_req_infos = SimpleNamespace(arr=np.full((2, 2, 2), -1, dtype=np.int64))

    module.fill_reqs_info(reqs)

    assert order == ["upload", "upload_waits_default", "default_waits_upload", "barrier"]
    assert [req.token_index.gpu_index_submitted_len for req in reqs] == [4, 8]
    if is_master_in_dp:
        assert module.shared_req_infos.arr[:, 0].tolist() == [[4, 1], [4, 0]]
    else:
        assert (module.shared_req_infos.arr == -1).all()
    assert cpu_table.is_pinned() and not cpu_table.is_shared()


def test_dp_cache_fetch_reserves_pages_and_keeps_logical_transfer_size(monkeypatch):
    table = torch.full((1, 16), -1, dtype=torch.int32)
    table[0, :4] = torch.arange(4, dtype=torch.int32)
    alloc_sizes = []
    next_index = [4]
    req = _bind_alloc_need(
        SimpleNamespace(
            req_idx=0,
            cur_kv_len=4,
            hold_kv_len=4,
            get_cur_total_len=lambda: 20,
        ),
        page_size=4,
    )
    source_table = torch.arange(32, dtype=torch.int32).reshape(2, 16)
    backend = SimpleNamespace(
        model=SimpleNamespace(mem_manager=SimpleNamespace()),
        dp_world_size=1,
        rank_in_dp=0,
        node_nccl_group=None,
        mem_managers=[SimpleNamespace(req_to_token_indexs=source_table)] * 2,
    )
    req.alloc_kv_mem = lambda size: _reserve_into_table(table, req, alloc_sizes, next_index, size)
    module = dp_shared_kv_trans.DPKVSharedMoudle.__new__(dp_shared_kv_trans.DPKVSharedMoudle)
    module.backend = backend
    module.dp_rank_in_node = 0
    module.shared_req_infos = SimpleNamespace(arr=np.array([[[4, 0], [10, 0]]], dtype=np.int64))
    context = SimpleNamespace(
        req_manager=make_cpu_index_manager(token_table=table, page_size=1),
        get_can_alloc_token_num=lambda: 100,
    )
    monkeypatch.setattr(dp_shared_kv_trans, "g_infer_context", context)
    monkeypatch.setattr(dp_shared_kv_trans.dist, "barrier", lambda group: None)

    tasks = module.build_shared_kv_trans_tasks([req], req_dp_ranks=[0])

    assert alloc_sizes == [8]
    assert req.cur_kv_len == 4
    assert req.hold_kv_len == 12
    assert len(tasks) == 1
    assert tasks[0].mem_indexes.tolist() == list(range(4, 10))
    assert len(tasks[0].max_kv_len_mem_indexes) == 6
