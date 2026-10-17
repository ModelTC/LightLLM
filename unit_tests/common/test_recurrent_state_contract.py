import pickle
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.recurrent_state import RecurrentState
from lightllm.common.recurrent_state_types import RecurrentStateContext
from lightllm.common.state_cache_manager import Glm5NextCacheConfig, LinearAttCacheConfig
from lightllm.server.core.objs.start_args_type import StartArgs


def test_published_context_is_detached_immutable_and_pickleable():
    tuning = {"BV": 8, "num_warps": 1}
    context = RecurrentStateContext.from_run_config(tuning)
    tuning.update(BV=128, num_warps=4, num_stages=1)
    assert dict(context.config) == {"BV": 8, "num_warps": 1, "num_stages": 3}
    with pytest.raises(FrozenInstanceError):
        context.config.BV = 128
    with pytest.raises(TypeError):
        context.config["BV"] = 128
    with pytest.raises(TypeError, match="from_run_config"):
        RecurrentStateContext(tuning)
    restored = pickle.loads(pickle.dumps(context))
    assert restored == context
    assert dict(restored.config) == dict(context.config)


def test_context_rejects_unrepresented_kernel_options():
    with pytest.raises(ValueError, match="Unknown SSM kernel configuration fields"):
        RecurrentStateContext.from_run_config({"BV": 32, "num_warps": 1, "future_layout_option": 2})
    with pytest.raises(TypeError, match="must be an integer"):
        RecurrentStateContext.from_run_config({"BV": [32], "num_warps": 1})


@pytest.fixture
def linear_config():
    return LinearAttCacheConfig(
        tp_world_size=1,
        full_att_all_num_kv_heads=1,
        full_att_dtype=torch.bfloat16,
        full_att_num_kv_heads=1,
        full_att_head_dim=32,
        global_linear_k_heads=1,
        global_linear_v_heads=2,
        num_linear_k_heads=1,
        num_linear_v_heads=2,
        head_linear_k_dim=32,
        head_linear_v_dim=32,
        conv_kernel_size=4,
        linear_layer_num=6,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
        full_attention_interval=3,
        all_layer_num=8,
    )


@pytest.mark.parametrize("layer,row", [(0, 0), (1, 1), (3, 2), (4, 3), (6, 4), (7, 5)])
def test_layer_buffers_follow_packed_linear_layers(linear_config, layer, row):
    owner = object.__new__(RecurrentState)
    owner.linear_config = linear_config
    conv = torch.arange(6).view(6, 1)
    ssm = conv + 10
    owner.conv_state = SimpleNamespace(buffer=conv)
    owner.ssm_state = SimpleNamespace(buffer=ssm)
    assert linear_config.get_linear_layer_index(layer) == row
    actual_conv, actual_ssm = owner.layer_buffers(layer)
    assert actual_conv.item() == row
    assert actual_ssm.item() == row + 10


@pytest.mark.parametrize("layer", [-1, 2, 5, 8])
def test_linear_mapping_rejects_full_attention_and_out_of_range_layers(linear_config, layer):
    with pytest.raises(AssertionError):
        linear_config.get_linear_layer_index(layer)


def test_model_configuration_retains_nondefault_kda_gate_bound():
    config = Glm5NextCacheConfig.from_model_config(
        {
            "kv_lora_rank": 512,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["deepseek_sparse_attention"],
            "index_kpool": 4,
            "index_head_dim": 128,
            "linear_attn_config": {
                "num_heads": 8,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "kda_layers": [0, 1, 2],
                "gate_lower_bound": -2.5,
            },
        },
        StartArgs(tp=4, data_type="bfloat16"),
    )
    assert config.gate_lower_bound == -2.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["compact", "replay"])
def test_kda_updater_receives_gate_bound_at_construction(linear_config, mode):
    linear_config.gate_lower_bound = -2.5
    args = SimpleNamespace(mtp_step=2, ssm_state_mode=mode, ssm_replay_config={})
    owner = RecurrentState(2, linear_config, args, kda=True)
    assert owner.updates.lower_bound == -2.5
