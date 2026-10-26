import pickle
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.linear_state import LinearStateManager, SSMRunConfig
from lightllm.common.state_cache_manager import Glm5NextCacheConfig, LinearAttCacheConfig
from lightllm.server.core.objs.start_args_type import StartArgs


def test_published_config_is_detached_immutable_and_pickleable():
    tuning = {"BV": 8, "num_warps": 1}
    config = SSMRunConfig(**tuning)
    tuning.update(BV=128, num_warps=4, num_stages=1)
    assert config == SSMRunConfig(BV=8, num_warps=1, num_stages=3)
    with pytest.raises(FrozenInstanceError):
        config.BV = 128
    restored = pickle.loads(pickle.dumps(config))
    assert restored == config


def test_config_rejects_unrepresented_kernel_options():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        SSMRunConfig(**{"BV": 32, "num_warps": 1, "future_layout_option": 2})
    with pytest.raises(TypeError, match="must be an integer"):
        SSMRunConfig(BV=[32], num_warps=1)


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
    owner = object.__new__(LinearStateManager)
    owner.linear_config = linear_config
    conv = torch.arange(6).view(6, 1)
    ssm = conv + 10
    owner.conv_state = SimpleNamespace(buffer=conv)
    owner.ssm_state = SimpleNamespace(buffer=ssm)
    assert linear_config.get_linear_layer_index(layer) == row
    actual_conv, actual_ssm = owner.get_layer_buffers(layer)
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
def test_kda_backend_receives_gate_bound_at_construction(linear_config, mode):
    linear_config.gate_lower_bound = -2.5
    args = SimpleNamespace(mtp_step=2, ssm_state_mode=mode, ssm_replay_config={})
    owner = LinearStateManager(2, linear_config, args, kda=True)
    assert owner.ssm_backend.gate_lower_bound == -2.5


@pytest.mark.parametrize("mode", ["compact", "replay", "parallel"])
def test_ssm_backend_publishes_detached_run_config(mode):
    from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import ReplaySSMBackend
    from lightllm.common.basemodel.triton_kernel.linear_att.compact_ssm import CompactSSMBackend

    ssm_backend = object.__new__(CompactSSMBackend if mode == "compact" else ReplaySSMBackend)
    tuning = {"BV": 32, "num_warps": 2}
    ssm_backend.run_config = tuning
    ssm_backend._config_is_fixed = True
    ssm_backend.compute_mode = "parallel" if mode == "parallel" else "inline"
    config = ssm_backend.get_run_config(*([None] * 7))
    assert isinstance(config, SSMRunConfig)
    tuning["BV"] = 128
    assert config.BV == 32
    assert config.num_stages == 3
    with pytest.raises(FrozenInstanceError):
        config.BV = 128


def test_compact_commits_each_unpadded_partition_with_its_verify_config(monkeypatch):
    from lightllm.common.basemodel.triton_kernel import mtp_utils
    from lightllm.common.basemodel.triton_kernel.linear_att.compact_ssm import CompactSSMBackend

    owner = object.__new__(LinearStateManager)
    owner.ssm_backend = object.__new__(CompactSSMBackend)
    owner.accepted_state_index = torch.tensor([0, 1, 2, 0], dtype=torch.int32)
    configs = [SSMRunConfig(8, 1), SSMRunConfig(32, 2)]
    commits = [owner.create_commit(config) for config in configs]
    owner.ssm_backend.run_config = {"BV": 128, "num_warps": 4}
    calls = []
    monkeypatch.setattr(mtp_utils, "linear_att_mtp_state_index_update", lambda **kwargs: None)
    monkeypatch.setattr(
        owner.ssm_backend,
        "accept_updates",
        lambda reqs, accepted, run_config: calls.append((reqs.tolist(), accepted, run_config)),
    )
    owner.commit_accepted_state(
        torch.tensor([0, 4, 8]),
        torch.tensor([2] * 4 + [0] * 4 + [1] * 4),
        torch.arange(4).repeat(3),
        torch.ones(12, dtype=torch.int32),
        4,
        linear_state_batches=((1, commits[0]), (0, None), (2, commits[1])),
    )
    assert [reqs for reqs, _, _ in calls] == [[2], [0, 1]]
    assert all(accepted is owner.accepted_state_index for _, accepted, _ in calls)
    assert [config for _, _, config in calls] == configs
