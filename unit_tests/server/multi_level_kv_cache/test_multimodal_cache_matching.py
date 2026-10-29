import time
from types import SimpleNamespace

import pytest
import torch

from lightllm.server.multi_level_kv_cache.manager import MultiLevelKVCacheManager
from lightllm.utils.kv_cache_utils import CpuKVCacheMeta


def test_disk_roundtrip_preserves_raw_pages_before_mm_boundary_trim(cpu_cache_factory, tmp_path, monkeypatch):
    light_mem = pytest.importorskip("light_mem")
    from lightllm.server.multi_level_kv_cache.disk_cache_worker import DiskCacheWorker

    client = cpu_cache_factory(CpuKVCacheMeta(3, 4096))
    pages, _ = client.allocate_pages([11, 22, 33], False)
    for index in pages:
        client.cpu_kv_cache_tensor[index].fill_(index + 1)
    client.update_pages_status_to_ready(pages, disk_offload_enable=True, token_num_in_page_list=768)
    expected = client.cpu_kv_cache_tensor.clone()
    worker = DiskCacheWorker.__new__(DiskCacheWorker)
    worker.cpu_cache_client = client
    worker.max_concurrent_write_tasks = 8
    worker.service = light_mem.PyLocalCacheService(
        worker._prepare_tensor(client.cpu_kv_cache_tensor),
        str(tmp_path / "cache"),
        storage_size=16 * 1024 * 1024,
        num_shard=1,
        num_worker=2,
    )
    for payloads in worker._gather_offload_payloads():
        worker._persist_pages_to_disk(payloads)
    deadline = time.monotonic() + 10
    while not all(worker.service.query([11, 22, 33])):
        assert time.monotonic() < deadline, "Disk write did not finish"
        time.sleep(0.01)

    # Evict CPU copies and overwrite their bytes before restoring from disk.
    replaced, _ = client.allocate_pages([44, 55, 66], False)
    client.recycle_pages(replaced)
    client.cpu_kv_cache_tensor.fill_(165)
    manager = MultiLevelKVCacheManager.__new__(MultiLevelKVCacheManager)
    manager.args = SimpleNamespace(enable_disk_cache=True)
    manager.cpu_cache_client = client
    manager.disk_cache_worker = worker
    loaded, disk_pages = manager._disk_cache_match([11, 22, 33], [])
    assert len(loaded) == disk_pages == 3
    assert torch.equal(client.cpu_kv_cache_tensor[loaded], expected)

    from lightllm.server.router.model_infer.mode_backend import multi_level_kv_cache as common

    copied_pages = []
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=0,
        hold_kv_len=0,
        mm_input_spans=[(600, 800)],
        sampling_param=SimpleNamespace(shm_param=SimpleNamespace(prompt_logprobs=-1)),
        shm_req=SimpleNamespace(
            input_len=1024,
            disk_prompt_cache_len=768,
            cpu_cache_match_page_indexes=SimpleNamespace(get_all=lambda: loaded),
            token_hash_page_len_list=SimpleNamespace(get_all=lambda: [256, 512, 768]),
        ),
    )

    def allocate(req, count):
        req.hold_kv_len = count
        return torch.arange(count, dtype=torch.int32)

    module = common.MultiLevelKvCacheModule.__new__(common.MultiLevelKvCacheModule)
    module.args = SimpleNamespace(page_size=1, cpu_cache_token_page_size=256)
    module.cpu_cache_client = client
    module.init_sync_group = None
    module.need_sync_compute_stream = lambda: False
    module.backend = SimpleNamespace(
        is_master_in_dp=True,
        is_hybrid_att_model=False,
        _alloc_req_kv_mem=allocate,
        model=SimpleNamespace(
            req_manager=SimpleNamespace(req_to_token_indexs=torch.arange(1024).view(1, -1)),
            att_resource_manager=SimpleNamespace(
                get_cpu_transfer_layouts=lambda: {},
                read_req_from_cpu=lambda buffer, pages, tokens, **kwargs: copied_pages.extend(pages),
            ),
        ),
    )
    monkeypatch.setattr(common.g_infer_context, "get_can_alloc_token_num", lambda: 1024)
    monkeypatch.setattr(common.dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None))
    module.load_cpu_cache_to_reqs([req])
    assert copied_pages == loaded[:2]
    assert req.cur_kv_len == req.hold_kv_len == req.shm_req.disk_prompt_cache_len == 512
    assert req.shm_req.cpu_prompt_cache_len == 0
    assert all(client.page_items.linked_items[page].ref_count == 0 for page in loaded)
