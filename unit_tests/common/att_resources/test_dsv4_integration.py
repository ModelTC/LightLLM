"""通过模型工厂和生产 fused kernel 验证 V4 State 接入。"""

import json
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.bounded_att import FP8E8M0SlidingWindowState
from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel
from lightllm.models.deepseek_v4.infer_struct import DeepseekV4InferStateInfo
from lightllm.utils.envs_utils import get_env_start_args


@pytest.fixture
def cache(monkeypatch):
    for key in ("LIGHTLLM_DP_WORLD_SIZE", "LIGHTLLM_CURRENT_RANK_IN_DP", "LIGHTLLM_CURRENT_DEVICE_ID"):
        monkeypatch.setenv(key, "1" if key.endswith("WORLD_SIZE") else "0")
    monkeypatch.setenv(
        "LIGHTLLM_START_ARGS",
        json.dumps(
            dict(
                page_size=256,
                pd_kv_page_size=256,
                run_mode="normal",
                enable_dp_prompt_cache_fetch=False,
                linear_att_hash_page_size=256,
                linear_att_page_block_num=8,
                cpu_cache_token_page_size=2048,
            )
        ),
    )
    get_env_start_args.cache_clear()
    config = dict(
        num_hidden_layers=3,
        compress_ratios=[4, 128, 0],
        head_dim=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        sliding_window=128,
        num_attention_heads=64,
    )
    args = SimpleNamespace(llm_kv_type="None", page_size=256, mtp_step=3)
    manager = AttentionStateManager(request_capacity=3)
    for state in DeepseekV4TpPartModel.create_full_att_states(
        model_config=config,
        args=args,
        model_logical_layer_ids=(0, 1, 2),
    ):
        state.device = torch.device("cuda")
        manager.register_state("main", state)
    bounded = DeepseekV4TpPartModel.create_bounded_att_states(
        model_config=config,
        args=args,
        model_logical_layer_ids=(0, 1, 2),
        request_capacity=3,
        device="cuda",
    )
    for state in bounded:
        manager.register_state("main", state)
    window = next(state for state in bounded if isinstance(state, FP8E8M0SlidingWindowState))
    manager.finish_state_registration()
    manager.allocate_storage(token_capacity=8192, page_size=256, big_page_tokens=2048, small_page_capacity=2)

    class TestModel(SimpleNamespace):
        __hash__ = object.__hash__

    model = TestModel(
        config=config,
        tp_rank_=0,
        tp_world_size_=1,
        max_req_num=2,
        batch_max_tokens=2048,
        layers_infer=[SimpleNamespace(layer_num_=i) for i in range(3)],
        att_state_role="main",
        args=SimpleNamespace(run_mode="normal", enable_prefill_microbatch_overlap=True),
    )
    from lightllm.common.basemodel.attention.nsa.dsv4_fp8_flashmla_sparse import DeepseekV4FlashMlaFp8SparseAttBackend

    backend = DeepseekV4FlashMlaFp8SparseAttBackend(model=model)
    yield manager, window, backend
    torch.cuda.synchronize()
    get_env_start_args.cache_clear()


def make_infer(manager, window):
    infer = DeepseekV4InferStateInfo()
    infer.att_state_manager = manager
    infer.att_state_role = "main"
    infer.first_logical_layer_id = window.config.logical_layer_ids[0]
    return infer


def read_packed_rows(buffer, slots, page_tokens):
    """独立参考读取 packed 数据/scale 分区，不复用 State 搬运算子。"""
    page_ids, rows = slots // page_tokens, slots % page_tokens
    data = rows[:, None] * 576 + torch.arange(576, device=slots.device)
    scale = page_tokens * 576 + rows[:, None] * 8 + torch.arange(8, device=slots.device)
    return torch.cat((buffer[page_ids[:, None], data], buffer[page_ids[:, None], scale]), dim=-1)


def test_backend_reads_owned_state_views(cache):
    manager, window, backend = cache
    infer = make_infer(manager, window)
    assert infer.swa_state is window
    infer.is_prefill = False
    for layer in range(3):
        swa, compressed = backend.get_att_input_params(infer, layer)
        assert swa.shape[1] == 74880
        assert (compressed is None) == (layer == 2)
        if layer != 2:
            assert compressed.data_ptr() == infer.get_compressed_kv_view(layer).kv.data_ptr()


@pytest.mark.parametrize("ratio,layer,is_indexer", [(4, 0, False), (4, 0, True), (128, 1, False)])
def test_real_compressor_continues_after_checkpoint_restore(cache, ratio, layer, is_indexer):
    from lightllm.models.deepseek_v4.layer_infer.compressor import fused_compress

    manager, window, _ = cache
    length, boundary = 1024, 512
    head_dim = 128 if is_indexer else 512
    scores = torch.randn((length, head_dim * (4 if ratio == 4 else 2)), device="cuda")
    cos = torch.ones((length, 32), device="cuda")
    sin, weight = torch.zeros_like(cos), torch.ones(head_dim, device="cuda")
    slots = [torch.arange(offset, offset + length, dtype=torch.int32, device="cuda") for offset in (0, 2048)]

    def compress(req_idx, start, end, decode=False):
        infer = make_infer(manager, window)
        positions = torch.arange(start, end, dtype=torch.int32, device="cuda")
        width = end - start
        infer.mem_index, infer.position_ids, infer.is_prefill = slots[req_idx][start:end], positions, not decode
        infer._dsv4_token_to_batch_idx = None if decode else torch.zeros(width, dtype=torch.int32, device="cuda")
        infer.b_req_idx = torch.full((width if decode else 1,), req_idx, dtype=torch.int32, device="cuda")
        infer.b_mtp_index = (
            torch.arange(width, dtype=torch.int32, device="cuda")
            if decode
            else torch.zeros(1, dtype=torch.int32, device="cuda")
        )
        infer.b_seq_len = positions + 1 if decode else torch.tensor([end], dtype=torch.int32, device="cuda")
        infer.b_ready_cache_len = None if decode else torch.tensor([start], dtype=torch.int32, device="cuda")
        infer.b_q_start_loc = None if decode else torch.zeros(1, dtype=torch.int32, device="cuda")
        return fused_compress(
            kv_score=scores[start:end],
            infer_state=infer,
            layer_idx=layer,
            norm_weight=weight,
            eps=1e-6,
            head_dim=head_dim,
            qk_rope_head_dim=64,
            compress_ratio=ratio,
            cos_table=cos,
            sin_table=sin,
            is_in_indexer=is_indexer,
        )

    reference = compress(0, 0, length)
    reference_kv = None if is_indexer else make_infer(manager, window).get_compressed_kv_view(layer).kv.clone()
    manager.reset_bounded_att_state([0, 1])
    compress(0, 0, boundary - 2)
    compress(0, boundary - 2, boundary)
    manager.save_bounded_att_runtime_to_pages([0], [boundary], "small", [0])
    manager.restore_bounded_att_runtime_from_pages([1], [boundary], "small", [0])
    for start, end in ((boundary, 768), (768, length - 4), (length - 4, length)):
        output = compress(1, start, end, decode=end == length)
        if is_indexer:
            closed = (torch.arange(start, end, device="cuda") + 1) % ratio == 0
            torch.testing.assert_close(output[closed], reference[start:end][closed], rtol=0, atol=0)
    if not is_indexer:
        actual = make_infer(manager, window).get_compressed_kv_view(layer).kv
        source = slots[0][boundary + ratio - 1 : length : ratio].long() // ratio
        target = slots[1][boundary + ratio - 1 : length : ratio].long() // ratio
        torch.testing.assert_close(
            read_packed_rows(reference_kv, source, 256 // ratio),
            read_packed_rows(actual, target, 256 // ratio),
            rtol=0,
            atol=0,
        )


def test_long_prefill_keeps_swa_prefix_and_runtime_tail(cache):
    from lightllm.models.deepseek_v4.layer_infer.transformer_layer_infer import DeepseekV4TransformerLayerInfer
    from lightllm.models.deepseek_v4.triton_kernel.norm_rope import fused_kv_norm_rope_to_swa
    from lightllm.models.deepseek_v4.triton_kernel.build_swa_indices import build_swa_indices

    manager, window, backend = cache
    reference = torch.zeros((17, 74880), dtype=torch.uint8, device="cuda")
    kv = torch.randn((2048, 512), dtype=torch.bfloat16, device="cuda")
    weight, angles = torch.randn(512, dtype=torch.bfloat16, device="cuda"), torch.randn((2048, 32), device="cuda")
    freqs = torch.complex(angles.cos(), angles.sin())
    layer = DeepseekV4TransformerLayerInfer.__new__(DeepseekV4TransformerLayerInfer)
    layer.layer_num_, layer.eps_, layer.freqs_cis = 2, 1e-6, freqs
    for start, end in ((0, 255), (255, 641), (641, 1100), (1100, 2048)):
        infer = make_infer(manager, window)
        positions = torch.arange(start, end, device="cuda", dtype=torch.int32)
        infer.is_prefill, infer.position_ids, infer.microbatch_index = True, positions, 0
        infer.b_req_idx = torch.zeros(1, dtype=torch.int32, device="cuda")
        infer.b_ready_cache_len, infer.b_seq_len = (
            torch.tensor([value], dtype=torch.int32, device="cuda") for value in (start, end)
        )
        prefill = backend.create_att_prefill_state(infer)
        prefill.prepare_swa_workspace(infer)
        infer.prefill_att_state = prefill
        indices = torch.empty((positions.numel(), 128), dtype=torch.int32, device="cuda")
        lengths, writes, runtime = (torch.empty_like(positions) for _ in range(3))
        build_swa_indices(
            token_req_idx=torch.zeros_like(positions),
            position_ids=positions,
            swa_state=window,
            swa_indices=indices,
            swa_lengths=lengths,
            swa_write_slots=writes,
            prefill_state=prefill,
            swa_runtime_write_slots=runtime,
            token_to_batch_idx=torch.zeros_like(positions),
            b_seq_len=infer.b_seq_len,
        )
        infer.dsv4_swa_write_slots, infer.dsv4_swa_runtime_write_slots = writes, runtime
        layer._norm_rope_and_cache_kv(kv[start:end], infer, SimpleNamespace(kv_norm_=SimpleNamespace(weight=weight)))
        fused_kv_norm_rope_to_swa(kv[start:end], weight, 1e-6, freqs, positions, positions, reference, 128)
        visible = torch.arange(max(0, start - 127), end, device="cuda")
        offset = prefill.swa_prefill_workspace_offsets[0] + visible - max(0, start - 127) // 128 * 128
        actual = read_packed_rows(prefill.swa_buffer[2].view(-1, 74880), offset.long(), 128)
        torch.testing.assert_close(actual[:, :-1], read_packed_rows(reference, visible, 128)[:, :-1], rtol=0, atol=0)
        tail = torch.arange(max(0, end - window.config.window_size), end, device="cuda")
        actual = read_packed_rows(window.swa_buffer[2, 0].view(-1, 74880), tail % window.swa_capacity_tokens, 128)
        torch.testing.assert_close(actual[:, :-1], read_packed_rows(reference, tail, 128)[:, :-1], rtol=0, atol=0)


@pytest.mark.parametrize("draft_kind", ["mtp", "dspark"])
def test_models_prepare_owned_states_in_one_manager(cache, draft_kind):
    from contextlib import nullcontext
    from lightllm.models.deepseek_v4_mtp.model import DeepseekV4MTPModel
    from lightllm.models.deepseek_v4_dspark.model import DeepseekV4DSparkModel

    _, _, backend = cache
    config = dict(
        backend.model.config,
        compress_ratios=[4, 128, 0, 0, 0],
        dspark_target_layer_ids=[0, 1],
        dspark_block_size=4,
        dspark_noise_token_id=1,
        dspark_markov_rank=4,
    )
    args = SimpleNamespace(
        llm_kv_type="None", page_size=256, mtp_step=3, mtp_mode="vanilla_with_att", run_mode="decode"
    )
    main = DeepseekV4TpPartModel.__new__(DeepseekV4TpPartModel)
    main.config, main.args, main.max_req_num, main.tp_rank_, main.tp_world_size_ = config, args, 2, 0, 1
    main.layers_infer = [SimpleNamespace(layer_num_=i) for i in range(3)]
    main.torch_memory_saver = SimpleNamespace(region=lambda **kwargs: nullcontext())
    main.prepare_attention_states()
    cls = DeepseekV4MTPModel if draft_kind == "mtp" else DeepseekV4DSparkModel
    drafts = []
    for index in range(2 if draft_kind == "mtp" else 1):
        draft = cls.__new__(cls)
        draft.main_model, draft.mtp_previous_draft_models = main, drafts.copy()
        draft.config = cls.normalize_model_config(config)
        assert draft.config == cls.normalize_model_config(draft.config)
        draft.args, draft.max_req_num, draft.tp_rank_, draft.tp_world_size_ = args, 2, 0, 1
        draft.torch_memory_saver = main.torch_memory_saver
        first_layer = 3 + sum(len(model.layers_infer) for model in drafts)
        draft.layers_infer = [
            SimpleNamespace(layer_num_=first_layer + i) for i in range(draft.config["num_hidden_layers"])
        ]
        draft.prepare_attention_states()
        drafts.append(draft)
    manager = main.att_state_manager
    assert all(model.att_state_manager is manager for model in drafts)
    manager.finish_state_registration()
    manager.allocate_storage(token_capacity=8192, page_size=256, big_page_tokens=2048, small_page_capacity=2)
    assert len(manager.get_full_attention_states()) == 2
    assert len(manager.get_bounded_attention_states()) == len(drafts) + 4
    for model in (main, *drafts):
        states = manager.get_bounded_attention_states(model.att_state_role)
        state = next(item for item in states if isinstance(item, FP8E8M0SlidingWindowState))
        assert state.config.logical_layer_ids == tuple(layer.layer_num_ for layer in model.layers_infer)
        assert state.swa_buffer.is_cuda
        if model is not main:
            assert len(states) == 1
