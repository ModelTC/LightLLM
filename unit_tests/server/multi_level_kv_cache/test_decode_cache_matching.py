"""验证初始化后的请求哈希直接用于 CPU / disk 匹配，不重复退页。"""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lightllm.server.core.objs.io_objs import GroupReqIndexes
from lightllm.server.core.objs.sampling_params import SamplingParams
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.multi_level_kv_cache import manager as manager_impl
from lightllm.utils.kv_cache_utils import CpuKVCacheMeta, compute_token_list_hash


@pytest.mark.parametrize("mode,hybrid", [("normal", True), ("prefill", True), ("decode", False), ("decode", True)])
@pytest.mark.parametrize("prefill_first", [False, True])
@pytest.mark.parametrize("input_len", [1, 256, 257, 258, 512, 513, 514, 769, 770])
@pytest.mark.parametrize("source", ["cpu", "disk", "partial_cpu"])
def test_only_unsafe_full_match_is_excluded_before_query(
    monkeypatch, cpu_cache_factory, local_req_memory, mode, hybrid, prefill_first, input_len, source
):
    client = cpu_cache_factory(CpuKVCacheMeta(3, 4096))
    req_args = StartArgs(
        run_mode=mode,
        enable_cpu_cache=True,
        cpu_cache_token_page_size=256,
        linear_att_hash_page_size=64,
        linear_att_page_block_num=4,
    )
    monkeypatch.setattr("lightllm.server.core.objs.req.get_env_start_args", lambda: req_args)
    monkeypatch.setattr("lightllm.server.core.objs.req.is_hybrid_att_model", lambda _: hybrid)
    params = SamplingParams()
    params.init(tokenizer=None, max_new_tokens=1)
    params.pd_prefill_first = prefill_first
    tokens = list(range(input_len))
    req = local_req_memory()
    req.init(1, tokens, params, None)
    allowed = req.token_hash_list.get_all()
    ends = req.token_hash_page_len_list.get_all()
    # 预先缓存完整候选，包括已被初始化退掉的末页，确保匹配器不会查询它。
    hashes = compute_token_list_hash(tokens, 256)
    hashes += [h for h in allowed if h not in hashes]
    cached = hashes if source == "cpu" else hashes[:1]
    pages, _ = client.allocate_pages(cached, False)
    client.update_pages_status_to_ready(pages)
    original_query = client.query_one_page
    query = Mock(side_effect=original_query)
    monkeypatch.setattr(client, "query_one_page", query)
    monkeypatch.setattr(manager_impl.zmq, "Context", Mock())
    monkeypatch.setattr(
        manager_impl, "get_shm_port_args", lambda: SimpleNamespace(multi_level_kv_cache_port=1, router_port=2)
    )
    monkeypatch.setattr(manager_impl, "CpuKvCacheClient", lambda **_: client)
    monkeypatch.setattr(manager_impl, "ShmReqManager", Mock())
    monkeypatch.setattr(manager_impl.threading.Thread, "start", lambda _: None)
    monkeypatch.setattr("lightllm.server.multi_level_kv_cache.disk_cache_worker.DiskCacheWorker", Mock())
    args = SimpleNamespace(
        run_mode=mode,
        model_dir="unused",
        zmq_mode="tcp://",
        enable_cpu_cache=True,
        enable_disk_cache=source == "disk",
        disk_cache_storage_size=1024,
        disk_cache_dir="unused",
        running_max_req_size=2,
        diverse_mode=False,
    )
    manager = manager_impl.MultiLevelKVCacheManager(args)
    matched = []
    manager.shm_req_manager.get_req_obj_by_index.return_value = req
    if source == "disk":
        manager.disk_cache_worker = SimpleNamespace(
            service=SimpleNamespace(_n=1),
            query_loadable_pages=Mock(side_effect=lambda hashs, start_pos: len(hashs) - start_pos),
            load_pages=Mock(return_value=True),
        )
    group = GroupReqIndexes(1, None, [0], time.time())
    expected = allowed[:1] if source == "partial_cpu" else allowed
    try:
        manager._handle_group_req_multi_cache_match(group, time.time())
        matched[:] = req.cpu_cache_match_page_indexes.get_all()
        assert len(matched) == len(expected)
        assert all(call.args[0] in allowed for call in query.call_args_list)
        if source == "disk":
            worker = manager.disk_cache_worker
            for call in worker.query_loadable_pages.call_args_list + worker.load_pages.call_args_list:
                assert call.kwargs["hashs"] == allowed
        cpu_count = min(len(cached), len(expected))
        expected_disk_len = (
            ends[len(expected) - 1] - (ends[cpu_count - 1] if cpu_count else 0)
            if source == "disk" and len(expected) > cpu_count
            else 0
        )
        assert req.disk_prompt_cache_len == expected_disk_len
        assert req.token_hash_list.get_all() == allowed
        assert manager.send_to_router_queue.get_nowait() is group
        manager.shm_req_manager.put_back_req_obj.assert_called_once_with(req)
        for page in pages:
            assert client.page_items.linked_items[page].ref_count == (1 if page in matched else 0)
        client.deref_pages(matched)
        assert all(client.page_items.linked_items[page].ref_count == 0 for page in matched)
    finally:
        manager.executor.shutdown(wait=False)
