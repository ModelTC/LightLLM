from types import SimpleNamespace

import numpy as np
import torch

from lightllm.models import get_model_class
from lightllm.models.deepseek_v41.model import DeepseekV41TpPartModel
from lightllm.server.router.model_infer.infer_batch import InferReq


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


def test_resumed_request_restores_only_the_ngram_predecessors():
    req = SimpleNamespace(req_idx=2, cur_kv_len=5, token_history_ready=False, get_input_token_ids=lambda: np.arange(8))
    assert InferReq.get_token_lookback(req, 3) == (2, 5, [2, 3, 4])
    assert InferReq.get_token_lookback(req, 3) is None
