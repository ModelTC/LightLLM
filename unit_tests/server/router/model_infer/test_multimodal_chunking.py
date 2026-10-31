from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("spans", [[], [(450, 600)], [(100, 800)], [(0, 384), (384, 768)]])
def test_chunk_ids_and_lengths_use_the_same_complete_boundaries(hybrid, spans):
    from lightllm.server.router.model_infer.infer_batch import InferReq

    req = InferReq.__new__(InferReq)
    req.args = SimpleNamespace(
        chunked_prefill_size=256,
        batch_max_tokens=1024,
        linear_att_hash_page_size=64,
        linear_att_page_block_num=8,
    )
    req.cur_kv_len = req.cur_output_len = 0
    req.mm_input_spans = spans
    req.hybrid_cache_len = 960
    req.shm_req = SimpleNamespace(input_len=1000, shm_prompt_ids=SimpleNamespace(arr=list(range(1000))))
    if hybrid:
        req.get_chuncked_input_token_len = req.get_chuncked_input_token_len_for_hybrid_att
        req.get_chuncked_input_token_ids = req.get_chuncked_input_token_ids_for_hybrid_att

    while req.cur_kv_len < req.get_cur_total_len():
        end = req.get_chuncked_input_token_len()
        assert req.get_chuncked_input_token_ids() == list(range(end))
        assert 0 < end - req.cur_kv_len <= req.args.batch_max_tokens
        assert not any(start < end < stop for start, stop in spans)
        req.cur_kv_len = end


@pytest.mark.usefixtures("kv_allocator_environment")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bounded transfers")
def test_bounded_reset_preserves_tail_and_cache_miss_resets_all(monkeypatch):
    from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context
    from unit_tests.common.att_resources.test_manager_helpers import composition, init_source

    monkeypatch.setattr(
        "lightllm.common.att_resources.bounded_att.utils.get_env_start_args",
        lambda: SimpleNamespace(enable_dp_prompt_cache_fetch=False),
    )
    resources = composition(device="cuda")
    slots = init_source(resources)
    tail = resources.main_kpool_c4_fp32_state
    resources.reset_bounded_att_state([1])
    assert (tail.tail_buffer[:, 1] == 9).all()
    resources.restore_bounded_att_pages(
        req_indices=[1],
        state_token_lens=[8],
        page_kind="big",
        slot_ids=[slots[-1]],
    )
    for state in resources.get_bounded_attention_states():
        state.mtp_state_index.fill_(2)

    monkeypatch.setattr(g_infer_context, "is_hybrid_att_model", True)
    monkeypatch.setattr(g_infer_context, "radix_cache", None)
    monkeypatch.setattr(g_infer_context, "att_resource_manager", resources)
    req = InferReq.__new__(InferReq)
    req.args = SimpleNamespace(
        linear_att_hash_page_size=4,
        linear_att_page_block_num=2,
        max_req_total_len=16,
        page_size=4,
    )
    req.req_idx = 1
    req.cur_kv_len = req.cur_output_len = 0
    req.sampling_param = SimpleNamespace(disable_prompt_cache=False)
    req.shm_req = SimpleNamespace(
        input_len=1, hybrid_token_hash_list=SimpleNamespace(get_all=lambda: [], size=0)
    )
    req._hybrid_match_radix_cache()

    for state in resources.get_bounded_attention_states():
        assert not state.conv_state[:, 1].any()
        assert not state.ssm_state[:, 3:6].any()
        assert state.mtp_state_index.tolist() == [2, 0, 2]
        assert state.conv_state[:, 0].any()
        pool = state.big_pages_pool
        expected_conv, expected_ssm = (23, 47) if state is resources.main_kda_conv_fp32_ssm_fp32_state else (24, 48)
        assert (pool.conv_state[slots[-1]] == expected_conv).all()
        assert (pool.ssm_state[slots[-1]] == expected_ssm).all()
    assert not tail.tail_buffer[:, 1].any()
    assert (tail.tail_buffer[:, 0] == 3).all()
    assert resources.bounded_att_big_page_allocator.free_count == 1
    assert req.cur_kv_len == req.hold_kv_len == req.shm_req.shm_cur_kv_len == 0


@pytest.mark.parametrize(
    "input_len,big_page_num,spans,expected_big,expected_small",
    [
        (2049, 8, [], [512, 1024, 1536, 2048], []),
        (2049, 8, [(450, 600)], [1024, 1536, 2048], []),
        (2049, 8, [(1100, 1700)], [512, 1024, 2048], []),
        (2049, 8, [(0, 700)], [1024, 1536, 2048], []),
        (2049, 8, [(450, 1700)], [2048], []),
        (2049, 8, [(1900, 2049)], [512, 1024, 1536], []),
        (2010, 8, [], [512, 1024, 1536], [1984]),
        (2010, 8, [(1900, 2000)], [512, 1024, 1536], []),
        (2010, 8, [(1900, 1984)], [512, 1024, 1536], [1984]),
        (2010, 10000000, [], [], [1984]),
        (2010, 10000000, [(1900, 2000)], [], []),
    ],
)
def test_hybrid_chunks_save_state_only_at_original_boundaries(
    input_len,
    big_page_num,
    spans,
    expected_big,
    expected_small,
):
    from lightllm.server.router.model_infer.infer_batch import InferReq, InferenceContext

    req = InferReq.__new__(InferReq)
    req.args = SimpleNamespace(
        chunked_prefill_size=256,
        batch_max_tokens=2048,
        linear_att_hash_page_size=64,
        linear_att_page_block_num=big_page_num,
        disable_chunked_prefill=False,
    )
    req.cur_kv_len = req.cur_output_len = req.req_idx = 0
    req.mm_input_spans = spans
    req.hybrid_cache_len = (input_len - 1) // 64 * 64
    req.hybrid_len_to_big_page_id = {}
    req.tail_small_page_buffer_id = None
    req.shm_req = SimpleNamespace(input_len=input_len)
    req.get_chuncked_input_token_len = req.get_chuncked_input_token_len_for_hybrid_att
    big_snapshots, small_snapshots = [], []
    page_ends = list(range(64 * big_page_num, req.hybrid_cache_len + 1, 64 * big_page_num))
    slots = iter(range(len(page_ends)))
    context = InferenceContext()
    context.args = req.args
    context.is_hybrid_att_model = True
    context.radix_cache = SimpleNamespace(free_one_small_page_buffer=lambda: None)
    context.att_resource_manager = SimpleNamespace(
        bounded_att_big_page_allocator=SimpleNamespace(alloc=lambda count: [next(slots)]),
        bounded_att_small_page_allocator=SimpleNamespace(alloc=lambda count: [7]),
        save_bounded_att_pages=lambda req_indices, state_token_lens, page_kind, slot_ids: (
            big_snapshots if page_kind == "big" else small_snapshots
        ).extend(state_token_lens),
    )

    while req.cur_kv_len < req.get_cur_total_len():
        context.save_hybrid_state_to_cache(torch.tensor([0]), [req])
        req.cur_kv_len = req.get_chuncked_input_token_len()

    assert big_snapshots == expected_big
    assert small_snapshots == expected_small
    assert sorted(req.hybrid_len_to_big_page_id) == page_ends
    assert len(set(req.hybrid_len_to_big_page_id.values())) == len(page_ends)
    assert big_snapshots == [end for end in page_ends if not any(start < end < stop for start, stop in spans)]


@pytest.mark.parametrize(
    "page_size,same_image,gpu_end,cpu_end",
    [(1, True, 1201, 1024), (1, False, 700, 512), (128, True, 0, 1024), (128, False, 640, 512)],
)
def test_full_attention_gpu_restore_stays_outside_image(
    monkeypatch,
    tmp_path,
    cpu_cache_factory,
    page_size,
    same_image,
    gpu_end,
    cpu_end,
):
    from lightllm.server.multi_level_kv_cache.manager import MultiLevelKVCacheManager
    from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache
    from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context
    from lightllm.utils import shm_utils
    from lightllm.utils.kv_cache_utils import CpuKVCacheMeta, compute_token_list_hash

    # Neo layout: text + <img> + independent virtual image IDs + </img> + text.
    prefix = [7] * 699 + [10]
    source = prefix + list(range(1000000, 1000500)) + [11] + [20] * 400
    image_start_id = 1000000 if same_image else 2000000
    query = prefix + list(range(image_start_id, image_start_id + 500)) + [11] + [21] * 400
    monkeypatch.setattr(shm_utils, "get_unique_server_name", lambda: tmp_path.name)
    radix_cache = RadixCache(4096, 0, page_size=page_size)
    radix_cache.insert(torch.tensor(source))

    monkeypatch.setattr(g_infer_context, "is_hybrid_att_model", False)
    monkeypatch.setattr(g_infer_context, "radix_cache", radix_cache)
    monkeypatch.setattr(
        g_infer_context,
        "req_manager",
        SimpleNamespace(req_to_token_indexs=torch.empty((1, len(query)), dtype=torch.int64)),
    )
    req = InferReq.__new__(InferReq)
    req.args = SimpleNamespace(page_size=page_size)
    req.sampling_param = SimpleNamespace(disable_prompt_cache=False)
    req.cur_kv_len = req.cur_output_len = req.hold_kv_len = req.req_idx = 0
    req.shared_kv_node = None
    req.mm_input_spans = [(700, 1200)]
    req.shm_req = SimpleNamespace(input_len=len(query), shm_prompt_ids=SimpleNamespace(arr=query))
    req._match_radix_cache()
    assert req.cur_kv_len == req.hold_kv_len == gpu_end
    assert not any(start < gpu_end < end for start, end in req.mm_input_spans)
    assert radix_cache.get_refed_tokens_num() == gpu_end

    hashes = compute_token_list_hash(source, 256)
    client = cpu_cache_factory(CpuKVCacheMeta(len(hashes), 4096))
    pages, _ = client.allocate_pages(hashes, False)
    client.update_pages_status_to_ready(pages)
    manager = MultiLevelKVCacheManager.__new__(MultiLevelKVCacheManager)
    manager.cpu_cache_client = client
    matched = manager._cpu_cache_match(compute_token_list_hash(query, 256))
    assert len(matched) * 256 == cpu_end
    # The page containing the image tail also contains the differing text.
    assert any(start < cpu_end < end for start, end in req.mm_input_spans) == same_image
    client.deref_pages(matched)
    radix_cache.dec_node_ref_counter(req.shared_kv_node)
    assert radix_cache.get_refed_tokens_num() == 0
    assert radix_cache.root_node.ref_counter == 1


@pytest.mark.parametrize("disable_chunked_mm_input", [False, True])
def test_mm_chunking_keeps_the_original_state_cache_position(monkeypatch, disable_chunked_mm_input):
    from lightllm.server.router.model_infer import infer_batch

    shm_req = SimpleNamespace(
        link_prompt_ids_shm_array=lambda: None,
        link_logprobs_shm_array=lambda: None,
        hybrid_token_hash_list=SimpleNamespace(size=3),
        input_len=1000,
        shm_prompt_ids=SimpleNamespace(arr=list(range(1000))),
    )
    monkeypatch.setattr(infer_batch.g_infer_context, "is_hybrid_att_model", True)
    monkeypatch.setattr(
        infer_batch.g_infer_context, "shm_req_manager", SimpleNamespace(get_req_obj_by_index=lambda index: shm_req)
    )
    monkeypatch.setattr(
        infer_batch.g_infer_context,
        "req_manager",
        SimpleNamespace(req_sampling_params_manager=SimpleNamespace(init_req_sampling_params=lambda req: None)),
    )
    monkeypatch.setattr(
        infer_batch,
        "InferSamplingParams",
        lambda *args: SimpleNamespace(
            pd_decode_node=None,
            shm_param=SimpleNamespace(stop_sequences=SimpleNamespace(to_token_ids_list=lambda: [])),
        ),
    )
    monkeypatch.setattr(infer_batch, "PromptSelectedLogprobsExt", lambda req: None)
    monkeypatch.setattr(infer_batch, "FinalTokenMetadataExt", lambda req: None)
    req = infer_batch.InferReq.__new__(infer_batch.InferReq)
    req.args = SimpleNamespace(
        linear_att_hash_page_size=256,
        linear_att_page_block_num=8,
        disable_chunked_mm_input=disable_chunked_mm_input,
        batch_max_tokens=1024,
    )
    req.shm_index, req.vocab_size = 0, 100
    req.multimodal_params = SimpleNamespace(
        to_dict=lambda: {
            "images": [
                {"start_idx": 450, "token_num": 150},
                {"start_idx": 700, "token_num": 200},
            ],
            "audios": [],
        }
    )
    req._init_all_state()
    assert req.hybrid_cache_len == 768


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize(
    "loaded_start,load_end,spans,disk_tokens,expected,cpu_tokens,remaining_disk_tokens",
    [
        (0, 0, [], 0, 0, 0, 0),
        (0, 4096, [], 2048, 4096, 2048, 2048),
        (0, 2048, [(2048, 2500)], 0, 2048, 2048, 0),
        (0, 4096, [(3500, 4096)], 2048, 4096, 2048, 2048),
        (0, 4096, [(3500, 4500)], 0, 2048, 2048, 0),
        (0, 4096, [(3500, 4500)], 2048, 2048, 2048, 0),
        (0, 4096, [(3500, 4500)], 4096, 2048, 0, 2048),
        (0, 4096, [(1000, 2500), (3500, 4500)], 2048, 0, 0, 0),
        (2304, 4096, [(3000, 4500)], 2048, 2304, 0, 0),
        (0, 6144, [(1000, 2500), (3500, 4500)], 2048, 6144, 4096, 2048),
        (2304, 6144, [(3000, 4500)], 2048, 6144, 1792, 2048),
    ],
)
def test_cpu_cache_match_and_load_use_a_complete_restore_point(
    monkeypatch,
    cpu_cache_factory,
    hybrid,
    loaded_start,
    load_end,
    spans,
    disk_tokens,
    expected,
    cpu_tokens,
    remaining_disk_tokens,
):
    from lightllm.server.router.model_infer.mode_backend import multi_level_kv_cache as common
    from lightllm.utils.kv_cache_utils import CpuKVCacheMeta

    ends = list(range(2048, load_end + 1, 2048))
    client = cpu_cache_factory(CpuKVCacheMeta(max(1, len(ends)), 16))
    page_ids, _ = client.allocate_pages(list(range(10, 10 + len(ends))), disk_offload_enable=False)
    client.update_pages_status_to_ready(page_ids, deref=False)
    loaded = []
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=loaded_start,
        hold_kv_len=loaded_start,
        mm_input_spans=spans,
        sampling_param=SimpleNamespace(shm_param=SimpleNamespace(prompt_logprobs=-1)),
        shm_req=SimpleNamespace(
            input_len=8192,
            disk_prompt_cache_len=disk_tokens,
            cpu_cache_match_page_indexes=SimpleNamespace(get_all=lambda: page_ids),
            token_hash_page_len_list=SimpleNamespace(get_all=lambda: ends),
        ),
    )
    table = torch.arange(8192, dtype=torch.int32).view(1, -1)

    def allocate(req, count):
        req.hold_kv_len += count
        return torch.arange(req.cur_kv_len, req.cur_kv_len + count, dtype=torch.int32)

    module = common.MultiLevelKvCacheModule.__new__(common.MultiLevelKvCacheModule)
    module.args = SimpleNamespace(page_size=256, cpu_cache_token_page_size=256)
    module.backend = SimpleNamespace(
        is_master_in_dp=True,
        is_hybrid_att_model=hybrid,
        _alloc_req_kv_mem=allocate,
        model=SimpleNamespace(
            req_manager=SimpleNamespace(req_to_token_indexs=table),
            att_resource_manager=SimpleNamespace(
                get_cpu_transfer_layouts=lambda: {},
                bounded_att_big_page_allocator=SimpleNamespace(alloc=lambda count: list(range(count))),
                read_req_from_cpu=lambda cpu_page_buffers, cpu_page_indices, token_indices, **kwargs: loaded.append(
                    req.hold_kv_len
                ),
            ),
        ),
    )
    module.cpu_cache_client = client
    module.init_sync_group = None
    module.need_sync_compute_stream = lambda: False
    monkeypatch.setattr(common.g_infer_context, "get_can_alloc_token_num", lambda: 8192)
    monkeypatch.setattr(common.g_infer_context, "is_hybrid_att_model", hybrid)
    monkeypatch.setattr(common.dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None))
    req.hybrid_len_to_big_page_id = {}
    module.load_cpu_cache_to_reqs([req])
    assert req.cur_kv_len == req.hold_kv_len == expected
    assert loaded == ([expected] if expected > loaded_start else [])
    assert req.shm_req.cpu_prompt_cache_len == cpu_tokens
    assert req.shm_req.disk_prompt_cache_len == remaining_disk_tokens
    assert all(client.page_items.linked_items[page].ref_count == 0 for page in page_ids)


@pytest.mark.parametrize("state_slot,expected", [(None, 1), (3, 2)])
def test_cpu_offload_requires_tail_state(monkeypatch, state_slot, expected):
    from lightllm.server.router.model_infer.mode_backend import multi_level_kv_cache as common

    monkeypatch.setattr(common.g_infer_context, "is_hybrid_att_model", True)
    module = common.MultiLevelKvCacheModule.__new__(common.MultiLevelKvCacheModule)
    module.args = SimpleNamespace(cpu_cache_token_page_size=512, disable_linear_att_small_page_cpu_cache=False)
    req = SimpleNamespace(tail_small_page_buffer_id=state_slot)
    assert module._handle_hybrid_att_last_page(req, 2, [512, 960]) == expected
