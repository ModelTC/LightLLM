from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.basemodel.attention.nsa.dsv4_fp8_flashmla_sparse import DeepseekV4FlashMlaFp8SparseAttBackend
from lightllm.models import get_model_class
from lightllm.models.deepseek_v41.infer_struct import DeepseekV41InferStateInfo
from lightllm.models.deepseek_v41.model import DeepseekV41TpPartModel
from lightllm.server.router.model_infer.infer_batch import InferReq, InferenceContext, g_infer_context


def test_consumers_share_global_kv_and_only_pair_producers_keep_pending_tokens():
    model_config = {
        "model_type": "deepseek_v41",
        "quantization_config": {"expert_dtype": "fp4"},
        "text_config": {
            "model_type": "deepseek_v41_text",
            "hidden_size": 512,
            "num_hidden_layers": 6,
            "num_attention_heads": 8,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "index_head_dim": 128,
            "sliding_window": 128,
            "compress_ratios": [0, 2, 2, 2, 1, 1],
            "kv_source_layer_ids": [1, 3, 4],
        },
    }
    model = get_model_class(model_config)
    assert model is DeepseekV41TpPartModel
    config = model.normalize_model_config(model_config)
    assert config["model_type"] == "deepseek_v41"
    assert config["expert_dtype"] == "fp4"
    args = SimpleNamespace(llm_kv_type="fp8kv_dsa", page_size=256, mtp_step=0)
    layers = tuple(range(config["num_hidden_layers"]))
    c1, c2 = model.create_full_att_states(config, args, layers)
    assert c2.config.logical_layer_groups == ((1, 2), (3,))
    assert c1.config.logical_layer_groups == ((4, 5),)
    for state in (c2, c1):
        assert state.config.block_token_size == 64 * state.config.compress_ratio
        assert state.config.indexer_dim == 128
    swa, pair = model.create_bounded_att_states(config, args, layers, request_capacity=2, device="cpu")
    assert swa.config.logical_layer_ids == layers
    assert pair.config.logical_layer_ids == (1, 3)
    assert pair.get_layer_view(1).shape == (2, 2, 1024)
    assert pair.buffer.dtype == torch.float32
    manager = AttentionStateManager(request_capacity=2)
    for state in (c1, c2, swa, pair):
        manager.register_state("main", state)
    manager.finish_state_registration()
    model_instance = model.__new__(model)
    model_instance.config = config
    model_instance.tp_world_size_ = 1
    model_instance.att_state_manager = manager
    backend = DeepseekV4FlashMlaFp8SparseAttBackend(model_instance)
    assert backend.swa_config is swa.config
    assert backend.compressed_configs[1] is c1.config
    assert backend.compressed_configs[2] is c2.config
    assert backend.compress_ratios == (0, 2, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_indexer_metadata_uses_c1_and_c2_page_layouts_and_masks_hold_rows(monkeypatch):
    infer = DeepseekV41InferStateInfo()
    infer.is_prefill = True
    infer.position_ids = torch.tensor([1, 129, 255, 0], dtype=torch.int32, device="cuda")
    infer.dsv4_sparse_req_idx = torch.tensor([0, 0, 1, 0], dtype=torch.int32, device="cuda")
    slots = torch.arange(512, 768, dtype=torch.int32, device="cuda")
    infer.req_manager = SimpleNamespace(
        req_to_token_indexs=torch.stack((slots, torch.zeros_like(slots))), HOLD_REQUEST_ID=1
    )

    def compressed_state(layer_id):
        ratio = 2 if layer_id in (1, 2) else 1
        return SimpleNamespace(config=SimpleNamespace(compress_ratio=ratio, block_token_size=64 * ratio))

    monkeypatch.setattr(infer, "get_compressed_kv_state", compressed_state)
    c2 = infer.get_indexer_metadata(1, 128)
    assert infer.get_indexer_metadata(2, 128) is c2
    assert c2.valid_lengths.tolist() == [1, 65, 0, 0]
    assert c2.context_lengths.flatten().tolist() == [1, 65, 1, 1]
    assert c2.row_page_table.tolist() == [[4, 0], [4, 5], [0, 0], [0, 0]]

    c1 = infer.get_indexer_metadata(3, 256)
    assert c1.valid_lengths.tolist() == [2, 130, 0, 1]
    assert c1.context_lengths.flatten().tolist() == [2, 130, 1, 1]
    assert c1.row_page_table.tolist() == [[8, 0, 0, 0], [8, 9, 10, 0], [0, 0, 0, 0], [8, 0, 0, 0]]


def test_cached_lookback_can_be_collected_after_kv_release(monkeypatch):
    monkeypatch.setattr(g_infer_context, "engram_lookback_depth", 3)
    context = InferenceContext()
    req = SimpleNamespace(
        req_idx=2,
        cur_kv_len=5,
        hold_kv_len=5,
        token_history_ready=False,
        get_input_token_ids=lambda: np.arange(8),
        token_index=SimpleNamespace(get_free_token_indexes=lambda start, end: [], reset=lambda: None),
        shm_req=SimpleNamespace(shm_cur_kv_len=5),
    )
    assert InferReq.get_token_lookback(req) == [(2, 2, 2), (2, 3, 3), (2, 4, 4)]
    assert not InferReq.get_token_lookback(req)

    context.free_a_req_mem([], req)
    req.cur_kv_len = 6
    assert InferReq.get_token_lookback(req) == [(2, 3, 3), (2, 4, 4), (2, 5, 5)]
