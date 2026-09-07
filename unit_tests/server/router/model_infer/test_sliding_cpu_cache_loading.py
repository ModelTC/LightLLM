from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.triton_kernel import sliding_window_cpu_cache_copy as copy_kernels
from lightllm.common.kv_cache_mem_manager.operator import hybrid_sliding as operator_module
from lightllm.common.req_manager.sliding_window import ReqManagerForSlidingWindow
from lightllm.server.router.model_infer.mode_backend import multi_level_kv_cache as cache_module


@pytest.mark.parametrize(
    "gpu_prefix,cpu_prefix,expected_endpoints",
    [(288, 736, {512: 0}), (288, 768, {512: 1, 768: 0}), (544, 736, {})],
)
def test_cpu_load_prepends_partial_gpu_page_and_restores_absolute_checkpoint(
    monkeypatch, gpu_prefix, cpu_prefix, expected_endpoints
):
    # Run the real public loader, sliding operator and runtime restore with
    # CPU tensors. Only the transfer kernel and CUDA/distributed calls are
    # replaced; GPU kernel and stream behavior have separate coverage.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, non_blocking=False: self)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(cache_module.dist, "barrier", lambda group: None)
    args = SimpleNamespace(cpu_cache_token_page_size=256, linear_att_hash_page_size=32, linear_att_page_block_num=8)
    monkeypatch.setattr(operator_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(operator_module, "get_current_rank_in_dp", lambda: 0)
    monkeypatch.setattr(operator_module, "get_dp_world_size", lambda: 1)

    allocated_tokens, evicted_tokens, dereferenced_pages, transfers = [], [], [], []
    states = torch.zeros((6, 1, 4, 2, 4))
    free_ids = iter(range(4))
    state_pool = SimpleNamespace(
        state_cache=states,
        alloc_one_state_cache=lambda: next(free_ids),
        get_state_cache=lambda index: states[index],
    )

    def alloc(need_size):
        allocated_tokens.append(need_size)
        return torch.arange(1000, 1000 + need_size, dtype=torch.int32)

    mem_manager = SimpleNamespace(
        alloc=alloc,
        sliding_config=object(),
        kv_buffer=object(),
        linear_att_big_page_buffers=state_pool,
        CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID=4,
    )
    mem_manager.operator = operator_module.HybridSlidingMemOperator(mem_manager)
    req_manager = object.__new__(ReqManagerForSlidingWindow)
    req_manager.mem_manager, req_manager.sliding_window = mem_manager, 4
    req_manager.req_to_sliding_window = torch.full((1, 8, 2, 4), -1.0)
    req_manager.req_to_token_indexs = torch.full((2, 1024), -1, dtype=torch.int32)
    req_manager.req_to_token_indexs[1, :gpu_prefix] = torch.arange(10000, 10000 + gpu_prefix, dtype=torch.int32)
    original_mapping = req_manager.req_to_token_indexs.clone()
    radix_cache = SimpleNamespace(
        free_radix_cache_to_get_enough_token=lambda need_token_num: evicted_tokens.append(need_token_num)
    )
    monkeypatch.setattr(cache_module.g_infer_context, "req_manager", req_manager)
    monkeypatch.setattr(cache_module.g_infer_context, "radix_cache", radix_cache)
    monkeypatch.setattr(cache_module.g_infer_context, "get_can_alloc_token_num", lambda: 2048)

    req = SimpleNamespace(
        req_idx=1,
        cur_kv_len=gpu_prefix,
        linear_att_len_to_big_page_id={},
        sampling_param=SimpleNamespace(shm_param=SimpleNamespace(prompt_logprobs=-1)),
        shm_req=SimpleNamespace(
            input_len=cpu_prefix + 1,
            disk_prompt_cache_len=0,
            cpu_cache_match_page_indexes=SimpleNamespace(get_all=lambda: [4, 8, 12]),
            token_hash_page_len_list=SimpleNamespace(get_all=lambda: [256, 512, cpu_prefix]),
        ),
    )

    def load(**kwargs):
        # cur_kv_len must already be the absolute CPU endpoint when the
        # operator assigns full-page checkpoints, not the old GPU hit length.
        assert req.cur_kv_len == cpu_prefix
        transfers.append(kwargs)
        for state_id, cpu_page in zip(kwargs["big_page_buffer_ids"], kwargs["page_indexes"]):
            kwargs["gpu_sliding_state"][state_id].fill_(cpu_page.item())

    monkeypatch.setattr(copy_kernels, "copy_cpu_cache_to_kv_buffer", load)
    module = object.__new__(cache_module.MultiLevelKvCacheModule)
    module.backend = SimpleNamespace(
        is_master_in_dp=True,
        radix_cache=radix_cache,
        model=SimpleNamespace(mem_manager=mem_manager, req_manager=req_manager),
    )
    module.need_sync_compute_stream = lambda: False
    module.init_sync_group = object()
    module.cpu_cache_client = SimpleNamespace(
        cpu_kv_cache_tensor=object(),
        lock=SimpleNamespace(acquire_sleep1ms=lambda: None, release=lambda: None),
        deref_pages=lambda page_list: dereferenced_pages.extend(page_list),
    )

    module.load_cpu_cache_to_reqs([req])

    need_tokens = cpu_prefix - gpu_prefix
    page_start = gpu_prefix // 256 * 256
    new_indexes = torch.arange(1000, 1000 + need_tokens, dtype=torch.int32)
    expected_transfer = torch.cat([original_mapping[1, page_start:gpu_prefix], new_indexes])
    padding = (-len(expected_transfer)) % 256
    assert allocated_tokens == evicted_tokens == [need_tokens]
    assert len(transfers) == 1
    assert transfers[0]["mem_indexes"].tolist() == expected_transfer.tolist() + [-1] * padding
    assert transfers[0]["page_indexes"].tolist() == [4, 8, 12][gpu_prefix // 256 :]
    assert req.linear_att_len_to_big_page_id == expected_endpoints
    assert 4 not in req.linear_att_len_to_big_page_id.values()
    if cpu_prefix % 256:
        assert transfers[0]["big_page_buffer_ids"][-1].item() == 4
    torch.testing.assert_close(req_manager.req_to_token_indexs[1, :gpu_prefix], original_mapping[1, :gpu_prefix])
    torch.testing.assert_close(req_manager.req_to_token_indexs[1, gpu_prefix:cpu_prefix], new_indexes)
    assert torch.all(req_manager.req_to_token_indexs[1, cpu_prefix:] == -1)
    assert torch.all(req_manager.req_to_sliding_window[:, :4] == -1)
    assert torch.all(req_manager.req_to_sliding_window[:, 4:8] == 12)
    assert req.shm_req.cpu_prompt_cache_len == need_tokens
    assert req.shm_req.shm_cur_kv_len == cpu_prefix
    # Dereference all matched pages, including the page already covered by
    # the GPU prefix and omitted from the actual transfer.
    assert dereferenced_pages == [4, 8, 12]
