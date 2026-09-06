from types import SimpleNamespace

import pytest
import torch


VISION_CONFIG = {
    "vision_n_layers": 32,
    "vision_patch_size": 14,
    "vision_downsample_ratio": 3,
    "vision_max_n_token": 384,
    "vision_min_pixels": 147456,
    "vision_max_wh_ratio": 8,
}


class _Tokenizer:
    def convert_tokens_to_ids(self, token):
        assert token == "<｜deepseek_image｜>"
        return 99

    def encode(self, prompt, **kwargs):
        return prompt


@pytest.mark.parametrize("prefix_len", range(8))
def test_tokenizer_uses_position_dependent_canonical_suffix(prefix_len):
    from lightllm.models.deepseek_v4.model import DeepSeekV4Tokenizer

    tokenizer = DeepSeekV4Tokenizer(_Tokenizer(), "/unused", VISION_CONFIG)
    image = SimpleNamespace(
        token_id=100_000,
        token_num=13,
        start_idx=None,
        block_start_idx=None,
        block_end_idx=None,
    )
    params = SimpleNamespace(images=[image])
    prompt = [7] * prefix_len + [99, 8]

    result = tokenizer.encode(prompt, params)

    cache_skip = prefix_len % 4
    assert result == [7] * prefix_len + list(range(image.token_id + cache_skip, image.token_id + 13)) + [8]
    assert image.block_start_idx == prefix_len
    assert image.block_end_idx == prefix_len + image.token_num - cache_skip
    assert image.start_idx == prefix_len + 3 - cache_skip
    assert image.start_idx % 4 == 3
    assert result[image.block_start_idx : image.block_end_idx] == list(
        range(image.token_id + cache_skip, image.token_id + image.token_num)
    )


def test_tokenizer_keeps_dsv4_on_one_dimensional_positions():
    from lightllm.models.deepseek_v4.model import DeepSeekV4Tokenizer

    tokenizer = DeepSeekV4Tokenizer(_Tokenizer(), "/unused", VISION_CONFIG)
    image = SimpleNamespace(image_h=64, image_w=64, grid_thwd=None)

    assert tokenizer.get_image_token_length(image) > 0
    assert image.grid_thwd is None


def test_chunk_boundary_moves_before_complete_image_block():
    from lightllm.server.router.model_infer.infer_batch import InferReq

    req = InferReq.__new__(InferReq)
    req.args = SimpleNamespace(chunked_prefill_size=8192)
    req.cur_kv_len = 0
    req.cur_output_len = 0
    req.shm_req = SimpleNamespace(input_len=10_000)
    req.image_block_spans = [(8100, 8400)]

    assert req._get_chunked_input_end() == 8100

    req.cur_kv_len = 8100
    assert req._get_chunked_input_end() == 10_000


def test_chunk_boundary_between_adjacent_images_never_splits_the_previous_image():
    from lightllm.server.router.model_infer.infer_batch import InferReq

    req = InferReq.__new__(InferReq)
    req.args = SimpleNamespace(chunked_prefill_size=640)
    req.cur_kv_len = 0
    req.cur_output_len = 0
    req.shm_req = SimpleNamespace(input_len=768)
    req.image_block_spans = [(0, 384), (384, 768)]

    assert req._get_chunked_input_end() == 384

    req.cur_kv_len = 384
    assert req._get_chunked_input_end() == 768


def test_atomic_image_block_can_exceed_chunk_size():
    from lightllm.server.router.model_infer.infer_batch import InferReq

    req = InferReq.__new__(InferReq)
    req.args = SimpleNamespace(chunked_prefill_size=256)
    req.cur_kv_len = 0
    req.cur_output_len = 0
    req.shm_req = SimpleNamespace(input_len=1000)
    req.image_block_spans = [(100, 800)]

    assert req._get_chunked_input_end() == 100

    req.cur_kv_len = 100
    assert req._get_chunked_input_end() == 800


def test_radix_match_retries_when_page_alignment_lands_in_earlier_image(monkeypatch):
    from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context

    class FakeRadixCache:
        def __init__(self):
            self.calls = []

        def match_prefix(self, key, update_refs=False):
            self.calls.append((len(key), update_refs))
            matched_len = min(len(key) // 256 * 256, 768)
            if update_refs:
                node = SimpleNamespace(node_prefix_total_len=matched_len)
                return node, matched_len, torch.arange(matched_len)
            return None, matched_len, None

    radix_cache = FakeRadixCache()
    monkeypatch.setattr(g_infer_context, "is_linear_att_mixed_model", False)
    monkeypatch.setattr(g_infer_context, "is_deepseek_v4", True)
    monkeypatch.setattr(g_infer_context, "radix_cache", radix_cache)
    monkeypatch.setattr(
        g_infer_context,
        "req_manager",
        SimpleNamespace(req_to_token_indexs=torch.empty((1, 1025), dtype=torch.int64)),
    )

    req = InferReq.__new__(InferReq)
    req.sampling_param = SimpleNamespace(disable_prompt_cache=False)
    req.cur_kv_len = 0
    req.cur_output_len = 0
    req.req_idx = 0
    req.image_block_spans = [(450, 600), (700, 900)]
    req.shared_kv_node = None
    req.shm_req = SimpleNamespace(
        input_len=1025,
        shm_prompt_ids=SimpleNamespace(arr=list(range(1025))),
        prompt_cache_len=0,
        shm_cur_kv_len=0,
    )

    req._match_radix_cache()

    assert radix_cache.calls == [
        (1024, False),
        (700, False),
        (450, False),
        (450, True),
    ]
    assert req.cur_kv_len == 256
    assert req.shm_req.prompt_cache_len == 256


def test_recover_swa_budget_includes_atomic_image_block(monkeypatch):
    from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context

    monkeypatch.setattr(
        g_infer_context,
        "req_manager",
        SimpleNamespace(sliding_window=128, get_prompt_cache_page_size=lambda: 256),
    )

    req = InferReq.__new__(InferReq)
    req.args = SimpleNamespace(disable_chunked_prefill=False, chunked_prefill_size=128)
    req.cur_kv_len = 0
    req.cur_output_len = 0
    req.shm_req = SimpleNamespace(input_len=2000)
    req.image_block_spans = [(500, 884)]
    req.dsv4_swa_page_size = 128
    req.dsv4_c4_page_size = 64
    req.dsv4_has_c128 = False

    assert req.get_dsv4_recover_need_page_and_slot_num() == (8, 8, 0)


@pytest.mark.parametrize(
    ("loaded_start", "load_end", "spans", "expected"),
    [
        (0, 4096, [], 4096),
        (0, 2048, [(2048, 2500)], 2048),
        (0, 4096, [(3500, 4096)], 4096),
        (0, 4096, [(3500, 4500)], 2048),
        (0, 4096, [(1000, 2500), (3500, 4500)], 0),
        (2304, 4096, [(3000, 4500)], 0),
    ],
)
def test_cpu_cache_load_end_never_splits_an_image(loaded_start, load_end, spans, expected):
    from lightllm.server.router.model_infer.mode_backend.dsv4_multi_level_kv_cache import (
        Dsv4MultiLevelKvCacheModule,
    )

    req = SimpleNamespace(image_block_spans=spans)

    assert Dsv4MultiLevelKvCacheModule._get_image_safe_load_end(req, loaded_start, load_end, 2048) == expected


def test_cpu_cache_rechecks_image_boundary_after_capacity_changes(monkeypatch):
    from lightllm.server.router.model_infer.mode_backend import (
        dsv4_multi_level_kv_cache as cache_module,
    )

    capacity_results = iter([6144, 4096])
    capacity_calls = []
    prepare_calls = []
    loaded_pages = []
    finish_calls = []

    def get_loadable_cpu_cache_end(*args):
        capacity_calls.append(args)
        return next(capacity_results)

    def prepare_cpu_cache_load(*, token_num, loaded_end):
        prepare_calls.append((token_num, loaded_end))
        return SimpleNamespace(mem_indexes=torch.arange(token_num, dtype=torch.int32))

    def load_cpu_cache_pages(*, page_indexes, **kwargs):
        loaded_pages.append(page_indexes.tolist())

    req_to_token_indexs = torch.full((1, 6144), -1, dtype=torch.int32)
    req_manager = SimpleNamespace(
        req_to_token_indexs=req_to_token_indexs,
        finish_cpu_cache_load=lambda req_idx, loaded_end: finish_calls.append((req_idx, loaded_end)),
    )
    mem_manager = SimpleNamespace(
        cpu_cache_layout=SimpleNamespace(token_page_size=2048),
        n_c4=False,
        n_c128=False,
        allocator=SimpleNamespace(can_use_mem_size=8192),
        swa_page_allocator=SimpleNamespace(can_use_mem_size=2),
        get_loadable_cpu_cache_end=get_loadable_cpu_cache_end,
        prepare_cpu_cache_load=prepare_cpu_cache_load,
        operator=SimpleNamespace(load_cpu_cache_pages=load_cpu_cache_pages),
        commit_cpu_cache_load_plan=lambda plan: None,
    )
    module = object.__new__(cache_module.Dsv4MultiLevelKvCacheModule)
    module.backend = SimpleNamespace(
        is_master_in_dp=True,
        radix_cache=None,
        model=SimpleNamespace(mem_manager=mem_manager, req_manager=req_manager),
    )
    module.cpu_cache_client = object()
    module.init_sync_group = None
    module._dsv4_store_sessions = {}

    req = SimpleNamespace(
        req_id=7,
        req_idx=0,
        cur_kv_len=0,
        image_block_spans=[(3500, 4500)],
        shm_req=SimpleNamespace(
            cpu_cache_match_page_indexes=SimpleNamespace(get_all=lambda: [10, 11, 12]),
            token_hash_page_len_list=SimpleNamespace(get_all=lambda: [2048, 4096, 6144]),
            disk_prompt_cache_len=2048,
            cpu_prompt_cache_len=0,
            shm_cur_kv_len=0,
        ),
    )

    real_tensor = torch.tensor
    monkeypatch.setattr(
        cache_module.torch,
        "tensor",
        lambda data, **kwargs: real_tensor(data, **{key: value for key, value in kwargs.items() if key != "device"}),
    )
    monkeypatch.setattr(cache_module.torch.cuda, "Event", lambda: SimpleNamespace(record=lambda: None))
    monkeypatch.setattr(cache_module.dist, "barrier", lambda group: None)
    monkeypatch.setattr(cache_module.g_infer_context, "get_can_alloc_token_num", lambda: 8192)
    monkeypatch.setattr(
        cache_module.g_infer_context,
        "get_can_alloc_dsv4_page_and_slot_num",
        lambda: (2, 0, 0),
    )

    module.load_cpu_cache_to_reqs([req])

    assert len(capacity_calls) == 2
    assert prepare_calls == [(2048, 2048)]
    assert loaded_pages == [[10]]
    assert finish_calls == [(0, 2048)]
    assert req.cur_kv_len == 2048
    assert req.shm_req.shm_cur_kv_len == 2048
    assert req.shm_req.cpu_prompt_cache_len == 2048
    assert req.shm_req.disk_prompt_cache_len == 0
    assert module._dsv4_store_sessions[7].leased_pages == [10, 11, 12]


def test_vision_model_allows_cpu_cache():
    from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel

    model = SimpleNamespace(
        load_way="HF",
        tp_world_size_=1,
        config={
            "num_attention_heads": 8,
            "o_groups": 8,
            "index_n_heads": 8,
            "vision_n_layers": 1,
        },
        args=SimpleNamespace(enable_cpu_cache=True),
    )

    DeepseekV4TpPartModel._verify_params(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("is_hash", [False, True])
def test_router_uses_bias_vl_only_for_image_tokens(is_hash):
    from lightllm.models.deepseek_v4.layer_infer.transformer_layer_infer import DeepseekV4TransformerLayerInfer

    router = DeepseekV4TransformerLayerInfer.__new__(DeepseekV4TransformerLayerInfer)
    router.has_vision = True
    router.is_hash = is_hash
    router.vocab_size = 32
    router.num_experts_per_tok = 6
    router.routed_scaling_factor = 1.5
    router.alloc_tensor = torch.empty

    torch.manual_seed(0)
    logits = torch.randn((2, 256), dtype=torch.float32, device="cuda")
    hash_table = torch.zeros((router.vocab_size, 6), dtype=torch.int64, device="cuda")
    hash_table[2] = torch.tensor([1, 4, 7, 10, 13, 16], device="cuda")
    text_bias = torch.zeros(256, dtype=torch.float32, device="cuda")
    text_bias[20:26] = torch.tensor([60.0, 55.0, 50.0, 45.0, 40.0, 35.0], device="cuda")
    vision_bias = torch.zeros(256, dtype=torch.float32, device="cuda")
    vision_bias[200:206] = torch.tensor([60.0, 55.0, 50.0, 45.0, 40.0, 35.0], device="cuda")
    layer_weight = SimpleNamespace(
        gate_tid2eid_=SimpleNamespace(weight=hash_table),
        gate_bias_=SimpleNamespace(weight=text_bias),
        gate_bias_vl_=SimpleNamespace(weight=vision_bias),
    )
    infer_state = SimpleNamespace(is_prefill=True, input_ids=torch.tensor([2, 100_000], device="cuda"))

    weights, indices = router._select_experts(logits, infer_state, layer_weight)

    scores = torch.sqrt(torch.nn.functional.softplus(logits))
    text_indices = hash_table[infer_state.input_ids[0]] if is_hash else (scores[0] + text_bias).topk(6).indices
    image_indices = (scores[1] + vision_bias).topk(6).indices
    expected_indices = torch.stack((text_indices, image_indices))
    torch.testing.assert_close(indices, expected_indices)
    expected = scores.gather(1, expected_indices)
    expected = expected / expected.sum(dim=-1, keepdim=True) * 1.5
    torch.testing.assert_close(weights, expected, rtol=2e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_build_image_visibility_scatter():
    from lightllm.models.deepseek_v4.triton_kernel.build_swa_index_dsv4 import build_image_visibility

    image_spans = torch.tensor([[0, 1, 4], [0, 7, 3], [1, 6, 3]], dtype=torch.int32, device="cuda")
    b_q_start_loc = torch.tensor([0, 6], dtype=torch.int32, device="cuda")
    b_ready_cache_len = torch.tensor([2, 5], dtype=torch.int32, device="cuda")
    b_q_seq_len = torch.tensor([6, 5], dtype=torch.int32, device="cuda")
    image_left = torch.zeros(11, dtype=torch.int32, device="cuda")
    image_right = torch.zeros(11, dtype=torch.int32, device="cuda")

    build_image_visibility(
        image_spans,
        b_q_start_loc,
        b_ready_cache_len,
        b_q_seq_len,
        image_left,
        image_right,
    )

    assert image_left.cpu().tolist() == [1, 2, 3, 0, 0, 0, 0, 0, 1, 2, 0]
    assert image_right.cpu().tolist() == [2, 1, 0, 0, 0, 2, 0, 2, 1, 0, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_swa_index_adds_bidirectional_image_visibility():
    from lightllm.models.deepseek_v4.triton_kernel.build_swa_index_dsv4 import build_swa_index

    positions = torch.tensor([5, 8, 10], dtype=torch.int32, device="cuda")
    req_idx = torch.zeros(3, dtype=torch.int32, device="cuda")
    req_to_token = torch.arange(20, dtype=torch.int32, device="cuda").unsqueeze(0)
    full_to_swa = torch.arange(100, 120, dtype=torch.int32, device="cuda")
    output = torch.empty((3, 8), dtype=torch.int32, device="cuda")
    lengths = torch.empty(3, dtype=torch.int32, device="cuda")
    image_left = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    image_right = torch.tensor([0, 2, 0], dtype=torch.int32, device="cuda")

    build_swa_index(
        req_idx,
        positions,
        req_to_token,
        full_to_swa,
        output,
        lengths,
        window=4,
        image_left=image_left,
        image_right=image_right,
    )

    assert output.cpu().tolist() == [
        [105, 104, 103, 102, -1, -1, -1, -1],
        [105, 106, 107, 108, 109, 110, -1, -1],
        [106, 107, 108, 109, 110, -1, -1, -1],
    ]
    assert lengths.cpu().tolist() == [4, 6, 5]
