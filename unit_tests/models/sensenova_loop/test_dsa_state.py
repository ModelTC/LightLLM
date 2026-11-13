from copy import deepcopy

import pytest
import torch

from lightllm.models.sensenova_loop.model import SenseNovaLoopTpPartModel
from lightllm.models.sensenova_loop.dsa_state import DsaIndexerState
from lightllm.common.attention_state.full_att import GQAFullAttentionState
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.fixture
def model_config():
    return {
        "model_type": "sensenova_loop",
        "loop_settings": {"start": 4, "end": 28, "eval_count": 2},
        "mhc": {"start": 4, "end": 28, "streams": 2, "iterations": 20},
        "dsa": {"num_heads": 8, "head_dim": 128, "topk": 2048},
        "dsa_export": {
            "schema": "sensenova_loop.dsa",
            "version": 1,
            "key_norm_eps": 1e-6,
            "indexer_rope": "qwen_partial_mrope",
        },
        "text_config": {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "full_attention_interval": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {"partial_rotary_factor": 0.25},
        },
    }


def complete_config(config):
    config["text_config"]["layer_types"] *= 8
    return config


def test_dsa_config_normalization_is_lossless_and_idempotent(model_config):
    model_config = complete_config(model_config)
    original = deepcopy(model_config)
    normalized = SenseNovaLoopTpPartModel.normalize_model_config(model_config)
    assert model_config == original
    assert normalized["num_hidden_layers"] == 56
    assert normalized["layer_types"].count("full_attention") == 14
    assert normalized["dsa"]["rotary_dim"] == 64
    assert SenseNovaLoopTpPartModel.normalize_model_config(normalized) == normalized


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_separate_indexer_storage_keeps_all_execution_layers_and_reduces_bytes(model_config, tp):
    model_config = complete_config(model_config)
    args = StartArgs(tp=tp, dp=1, data_type="bf16", llm_kv_type="None")
    states = SenseNovaLoopTpPartModel.create_full_att_states(model_config, args, tuple(range(56)), 0, tp)
    assert len(states) == 2
    gqa, indexer = states
    assert type(gqa) is GQAFullAttentionState and type(indexer) is DsaIndexerState
    assert indexer.config.logical_layer_groups == gqa.config.logical_layer_groups
    assert len(indexer.config.logical_layer_groups) == 14
    assert indexer.get_state_type() == "dsa_indexer_bf16_d128"
    assert indexer.config.latent_dim == 128
    for state in states:
        state.device = torch.device("cpu")
        state.allocate_storage(16)
    assert indexer.get_layer_view(3).shape == (17, 1, 128)
    assert gqa.get_layer_view(3)[0].shape == (17, 4 // tp, 256)
    baseline_bytes = 14 * (4 // tp) * 2 * 384 * 2
    assert sum(state.get_token_bytes() for state in states) < baseline_bytes
    assert indexer.get_storage_layer_index(7) != indexer.get_storage_layer_index(31)


def test_rejects_quantized_dsa_before_allocating_gpu_storage(model_config):
    model_config = complete_config(model_config)
    args = StartArgs(tp=2, dp=1, data_type="bf16", llm_kv_type="fp8kv_sph")
    with pytest.raises(ValueError, match="unquantized KV"):
        SenseNovaLoopTpPartModel.create_full_att_states(model_config, args, tuple(range(56)), 0, 2)


def test_dense_sensenova_preserves_one_gqa_resource(model_config):
    model_config = complete_config(model_config)
    del model_config["dsa"]
    del model_config["dsa_export"]
    args = StartArgs(tp=2, dp=1, data_type="bf16", llm_kv_type="None")
    states = SenseNovaLoopTpPartModel.create_full_att_states(model_config, args, tuple(range(56)), 0, 2)
    assert len(states) == 1 and type(states[0]) is GQAFullAttentionState


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("target_rank", [0, 1, 2, 3])
def test_indexer_tp2_to_tp4_pd_and_cpu_cache_preserve_raw_bytes(target_rank, monkeypatch):
    from dataclasses import replace
    from lightllm.common.attention_state.full_att import MLAConfig
    from lightllm.common.attention_state.full_att.mla import mla_state
    from lightllm.common.attention_state.types import StateTransferLayout

    args = StartArgs(pd_kv_page_size=4, cpu_cache_token_page_size=4)
    monkeypatch.setattr(mla_state, "get_env_start_args", lambda: args)
    config = MLAConfig(((3,), (7,)), 128, torch.bfloat16, tp_rank=0, tp_world_size=2)
    source = DsaIndexerState(config, device="cuda")
    target = DsaIndexerState(replace(config, tp_rank=target_rank, tp_world_size=4), device="cuda")
    for state in (source, target):
        state.allocate_storage(16)
    source.storage.view(torch.uint8).random_(0, 256)
    target.storage.view(torch.uint8).fill_(173)
    layouts = []
    for state in (source, target):
        regions = tuple(
            replace(r, byte_start=r.byte_start + 32, byte_end=r.byte_end + 32) for r in state.get_pd_transfer_regions()
        )
        layouts.append(
            StateTransferLayout(
                state_name=state.get_state_type(),
                tokens_per_page=4,
                byte_start=32,
                byte_end=max(r.byte_end for r in regions),
                regions=regions,
            )
        )
    src = torch.tensor([6, 99, 3, 99, 10, 99], dtype=torch.int32, device="cuda")[::2]
    dst = torch.tensor([2, 8, 5], dtype=torch.int64, device="cuda")
    expected = source.storage[:, src].contiguous().view(torch.uint8)
    pages = torch.full((3, layouts[0].byte_end + 64), 91, dtype=torch.uint8, device="cuda")
    source.write_to_pd_page(src, pages, 1, layouts[0])
    target.read_from_pd_page(pages, dst, 1, layouts[1])
    assert torch.equal(target.storage[:, dst].contiguous().view(torch.uint8), expected)
    assert (pages[[0, 2]] == 91).all() and (pages[:, :32] == 91).all()
    # Every decode TP rank reads the full key; only rank zero may write it out.
    if target_rank:
        pages.fill_(91)
        target.write_to_pd_page(dst, pages, 2, layouts[1])
        assert (pages == 91).all()
    cpu = torch.full((3, layouts[0].byte_end + 64), 91, dtype=torch.uint8, pin_memory=True)
    source.write_to_cpu(src, cpu, [1], layouts[0])
    target.storage.view(torch.uint8).fill_(173)
    target.read_from_cpu(cpu, [1], dst, layouts[1])
    torch.cuda.synchronize()
    assert torch.equal(target.storage[:, dst].contiguous().view(torch.uint8), expected)
    untouched = [i for i in range(17) if i not in (2, 8, 5)]
    assert (target.storage[:, untouched].contiguous().view(torch.uint8) == 173).all()
