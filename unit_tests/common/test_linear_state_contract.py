import pickle
from dataclasses import FrozenInstanceError

import pytest
import torch

from lightllm.common.ssm_types import SSMRunConfig
from lightllm.common.attention_state.commit import StateAcceptance, StateCommit, StateCommitBatch
from lightllm.common.attention_state.bounded_att import KDAState
from unit_tests.common.test_linear_state import make_state
from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig, GDNState
from lightllm.models.glm5_next.config import Glm5NextResourceConfig
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
    return ConvSSMStateConfig(
        logical_layer_ids=(0, 1, 3, 4, 6, 7),
        global_k_heads=1,
        global_v_heads=2,
        key_dim=32,
        value_dim=32,
        conv_width=3,
        conv_dtype=torch.bfloat16,
        ssm_dtype=torch.float32,
    )


@pytest.mark.parametrize("layer,row", [(0, 0), (1, 1), (3, 2), (4, 3), (6, 4), (7, 5)])
def test_layer_buffers_follow_packed_linear_layers(linear_config, layer, row):
    owner = GDNState(linear_config, request_capacity=1, device="cpu")
    for i in range(6):
        owner.conv_state[i].fill_(i)
        owner.ssm_state[i].fill_(i + 10)
    actual_conv, actual_ssm = owner.get_layer_view(layer)
    assert (actual_conv == row).all()
    assert (actual_ssm == row + 10).all()


@pytest.mark.parametrize("layer", [-1, 2, 5, 8])
def test_linear_mapping_rejects_full_attention_and_out_of_range_layers(linear_config, layer):
    owner = GDNState(linear_config, request_capacity=1, device="cpu")
    with pytest.raises(KeyError):
        owner.get_layer_view(layer)


def test_model_configuration_retains_nondefault_kda_gate_bound():
    config = Glm5NextResourceConfig.from_model_config(
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


def test_kda_backend_receives_gate_bound_at_construction(linear_config):
    owner = KDAState(linear_config, 3, mtp_step=2, ssm_state_mode="replay", gate_lower_bound=-2.5, device="cpu")
    assert owner._ssm_executor.gate_lower_bound == -2.5


def test_replay_backend_publishes_detached_run_config():
    from lightllm.common.attention_state.bounded_att.ssm.replay import ReplaySSMExecutor

    ssm_backend = object.__new__(ReplaySSMExecutor)
    tuning = {"BV": 32, "num_warps": 2}
    ssm_backend.run_config = tuning
    ssm_backend._config_is_fixed = True
    ssm_backend.compute_mode = "inline"
    config = ssm_backend.get_run_config(*([None] * 7))
    assert isinstance(config, SSMRunConfig)
    tuning["BV"] = 128
    assert config.BV == 32
    assert config.num_stages == 3
    with pytest.raises(FrozenInstanceError):
        config.BV = 128


@pytest.mark.parametrize("cls", [GDNState, KDAState])
@pytest.mark.parametrize("mtp_step,expected", [(0, "inline"), (2, "precompute")])
def test_replay_defaults_to_precompute_for_speculative_decode(linear_config, cls, mtp_step, expected):
    owner = cls(linear_config, request_capacity=2, mtp_step=mtp_step, ssm_state_mode="replay", device="cpu")
    assert owner._ssm_executor.compute_mode == expected


def make_acceptance(batches=()):
    return StateAcceptance(torch.tensor([2, 0, 1]), torch.tensor([1, 2, 3]), batches)


def test_commit_partitions_and_resource_identity_are_validated():
    with pytest.raises(ValueError, match="nonnegative"):
        StateCommitBatch(-1)
    commit = StateCommit("main", SSMRunConfig(8, 1))
    with pytest.raises(ValueError, match="duplicate"):
        StateCommitBatch(1, (commit, commit))
    acceptance = make_acceptance((StateCommitBatch(3, (commit,)),))
    with pytest.raises(ValueError, match="unknown state"):
        acceptance.validate({"draft0"})
    with pytest.raises(ValueError, match="layout"):
        StateAcceptance(torch.tensor([1, 2]), torch.tensor([1])).validate()
    with pytest.raises(ValueError, match="partition"):
        make_acceptance((StateCommitBatch(2),)).validate({"main"})


@pytest.mark.parametrize("cls", [GDNState, KDAState])
def test_native_accept_restore_and_reuse_preserve_other_requests(cls):
    owner, _, _ = make_state(cls, step=2, device="cpu")
    owner.conv_state.copy_(torch.arange(5))
    for slot in range(owner.ssm_state.shape[1]):
        owner.ssm_state[:, slot].fill_(slot)
    before = owner.ssm_state.clone()
    owner.commit_bounded_att_state(StateAcceptance(torch.tensor([1]), torch.tensor([3], dtype=torch.int32)))
    accepted = owner.get_accepted_state(1)
    assert torch.equal(accepted.conv[0, 0], torch.tensor([2, 3, 4], dtype=accepted.conv.dtype))
    assert torch.all(accepted.ssm == 5)
    owner.restore(2, *accepted)
    restored = owner.get_accepted_state(2)
    assert torch.equal(restored.conv, accepted.conv)
    assert torch.equal(restored.ssm, accepted.ssm)
    owner.reset_bounded_att_state([2])
    assert torch.equal(owner.ssm_state[:, :6], before[:, :6])
    assert torch.equal(owner.ssm_state[:, 9:], before[:, 9:])
    assert not owner.ssm_state[:, 6:9].any()
    assert owner.mtp_state_index.tolist() == [0, 2, 0, 0]


@pytest.mark.parametrize("cls", [GDNState, KDAState])
def test_replay_fused_acceptance_keeps_cursors_inside_mode(monkeypatch, cls):
    from lightllm.common.attention_state.bounded_att.ssm import base

    owner, _, _ = make_state(cls, mode="replay", step=3, device="cpu")
    calls = []
    monkeypatch.setattr(base, "commit_ssm_prefix", lambda *args: calls.append(args))
    monkeypatch.setattr(owner._ssm_executor, "accept_updates", lambda *a: pytest.fail("unfused acceptance"))
    owner.commit_bounded_att_state(make_acceptance())
    assert len(calls) == 1
    assert calls[0][3] is owner._ssm_executor.cursors
    assert calls[0][0] is owner.mtp_state_index


@pytest.mark.parametrize("change", ["owner", "executor", "shape", "optional", "dtype"])
def test_graph_layout_mismatch_rejected_before_copy(change):
    from copy import copy

    owner, _, _ = make_state(mode="native", device="cpu")
    target = owner.prepare_decode(torch.tensor([0, 1], dtype=torch.int32), torch.tensor([0, 3, 6]))
    source = owner.prepare_decode(torch.tensor([1, 2], dtype=torch.int32), torch.tensor([0, 2, 5]))
    if change == "owner":
        source.resource = copy(owner)
    elif change == "executor":
        source.executor = copy(owner._ssm_executor)
    elif change == "shape":
        source.state_indices = source.state_indices[:1]
    elif change == "optional":
        source.accepted_token_counts = None
    else:
        source.cu_seqlens = source.cu_seqlens.int()
    with pytest.raises(ValueError, match="resource|layout"):
        target.copy_inputs(source)
    assert target.req_indices.tolist() == [0, 1]
    assert target.cu_seqlens.tolist() == [0, 3, 6]


@pytest.mark.parametrize("kda", [False, True])
@pytest.mark.parametrize("speculative", [False, True])
def test_native_forward_preserves_algorithm_inputs(monkeypatch, kda, speculative):
    from importlib import import_module

    owner, _, _ = make_state(KDAState if kda else GDNState, step=2 if speculative else 0, device="cpu")
    module, function = (
        ("fla.ops.kda_decode", "fused_recurrent_kda")
        if kda
        else ("mtp_fused_recurrent", "mtp_fused_recurrent_gated_delta_rule")
        if speculative
        else ("fla.ops", "fused_recurrent_gated_delta_rule")
    )
    calls = []

    def recurrent(**kwargs):
        calls.append(kwargs)
        return kwargs["v"], None

    monkeypatch.setattr(
        import_module("lightllm.common.basemodel.triton_kernel.linear_att." + module), function, recurrent
    )
    reqs = torch.tensor([1, 2], dtype=torch.int32)
    cu = torch.tensor([0, 3, 6], dtype=torch.int32) if speculative else None
    forward = owner.prepare_decode(reqs, cu)
    shape = (1, 6) if speculative else (2, 1)
    c = owner.config
    q = torch.zeros(*shape, c.num_k_heads, c.key_dim)
    v = torch.zeros(*shape, c.num_v_heads, c.value_dim)
    gate = torch.zeros(q.numel() // c.key_dim, c.num_v_heads * (c.key_dim if kda else 1))
    beta = torch.zeros(gate.shape[0], c.num_v_heads)
    a_log, bias = torch.zeros(c.num_v_heads), torch.zeros(gate.shape[-1])
    assert forward.run_layer(1, q, q, v, gate, beta, a_log, bias) is v
    call = calls[0]
    assert call["initial_state"].data_ptr() == owner.ssm_state[1].data_ptr()
    assert call["ssm_state_indices"] is forward.state_indices
    if speculative:
        assert torch.equal(call["cu_seqlens"], cu)
        assert call["num_accepted_tokens"] is forward.accepted_token_counts
    if kda:
        assert call["raw_gate"].shape == (*shape, c.num_v_heads * c.key_dim)
        assert call["raw_beta"].shape == (*shape, c.num_v_heads)
        assert call["lower_bound"] == -5.0
    else:
        assert call["a_raw"] is gate and call["b_raw"] is beta
    assert forward.finish() is None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_acceptance_keeps_request_order_with_strided_inputs(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    owner, _, _ = make_state(mode="replay", step=3, device=device)
    owner.mtp_state_index.fill_(2)
    owner._ssm_executor.cursors.fill_(8)
    reqs = torch.tensor([2, -99, 0, -99, 3, -99], device=device)[::2]
    lengths = torch.tensor([1, -99, 3, -99, 2, -99], dtype=torch.int32, device=device)[::2]
    owner.commit_bounded_att_state(StateAcceptance(reqs, lengths))
    assert owner.mtp_state_index.tolist() == [2, 2, 0, 1]
    assert owner._ssm_executor.cursors.tolist() == [11, 8, 9, 8]
