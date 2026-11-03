import pickle
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.ssm_types import SSMRunConfig
from lightllm.common.attention_state.commit import MtpCommitContext, StateCommit, StateCommitBatch
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


@pytest.mark.parametrize("mode", ["compact", "replay"])
def test_kda_backend_receives_gate_bound_at_construction(linear_config, mode):
    owner = KDAState(linear_config, 3, mtp_step=2, ssm_state_mode=mode, gate_lower_bound=-2.5, device="cpu")
    assert owner._ssm.gate_lower_bound == -2.5


@pytest.mark.parametrize("mode", ["compact", "replay", "parallel"])
def test_ssm_backend_publishes_detached_run_config(mode):
    from lightllm.common.attention_state.bounded_att.ssm.replay import ReplaySSMState
    from lightllm.common.attention_state.bounded_att.ssm.compact import CompactSSMState

    ssm_backend = object.__new__(CompactSSMState if mode == "compact" else ReplaySSMState)
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


def make_commit_context(batches):
    return MtpCommitContext(
        torch.tensor([0, 4, 8]),
        torch.tensor([2] * 4 + [0] * 4 + [1] * 4),
        torch.arange(4).repeat(3),
        torch.ones(12, dtype=torch.int32),
        4,
        batches,
    )


def test_compact_commits_each_unpadded_partition_with_its_verify_config(monkeypatch):
    from lightllm.common.attention_state.bounded_att.ssm import base

    owner, _, _ = make_state(mode="compact", step=3, device="cpu")
    reqs = torch.tensor([2, 0, 1], dtype=torch.int32)
    configs = [SSMRunConfig(8, 1), SSMRunConfig(32, 2)]
    commits = []
    for subset, config in zip((reqs[:1], reqs[1:]), configs):
        forward = owner.prepare_decode(subset, torch.arange(subset.numel() + 1) * 4)
        forward.run_config = config
        commits.append(forward.finish())
    owner._ssm.run_config = {"BV": 128, "num_warps": 4}
    calls = []
    monkeypatch.setattr(base, "linear_att_mtp_state_index_update", lambda **kwargs: None)
    monkeypatch.setattr(
        owner._ssm,
        "accept_updates",
        lambda reqs, accepted, run_config: calls.append((reqs.tolist(), accepted, run_config)),
    )
    context = make_commit_context(
        (StateCommitBatch(1, (commits[0],)), StateCommitBatch(0), StateCommitBatch(2, (commits[1],)))
    )
    owner.commit_bounded_att_state(reqs, torch.tensor([1, 2, 3]), context=context)
    assert [reqs for reqs, _, _ in calls] == [[2], [0, 1]]
    assert all(accepted is owner.mtp_state_index for _, accepted, _ in calls)
    assert [config for _, _, config in calls] == configs


@pytest.mark.parametrize("failure", ["missing", "wrong_owner", "mutable_config", "no_context"])
def test_invalid_compact_receipt_rejected_before_state_changes(monkeypatch, failure):
    owner, _, _ = make_state(mode="compact", step=3, device="cpu")
    owner.mtp_state_index.fill_(2)
    name = "draft0_other_state" if failure == "wrong_owner" else owner.state_name
    config = {"BV": 8, "num_warps": 1} if failure == "mutable_config" else SSMRunConfig(8, 1)
    commits = () if failure == "missing" else (StateCommit(name, config),)
    context = None if failure == "no_context" else make_commit_context((StateCommitBatch(3, commits),))
    with pytest.raises(ValueError, match="compact"):
        owner.commit_bounded_att_state(torch.tensor([2, 0, 1]), torch.tensor([1, 2, 3]), context=context)
    assert owner.mtp_state_index.tolist() == [2] * 4


def test_commit_partitions_and_resource_identity_are_validated():
    with pytest.raises(ValueError, match="nonnegative"):
        StateCommitBatch(-1)
    commit = StateCommit("main", SSMRunConfig(8, 1))
    with pytest.raises(ValueError, match="duplicate"):
        StateCommitBatch(1, (commit, commit))
    context = make_commit_context((StateCommitBatch(3, (commit,)),))
    with pytest.raises(ValueError, match="unknown state"):
        context.validate(3, {"draft0"})
    with pytest.raises(ValueError, match="counts differ"):
        context.validate(2, {"main"})
    with pytest.raises(ValueError, match="partition"):
        make_commit_context((StateCommitBatch(2),)).validate(3, {"main"})


@pytest.mark.parametrize("cls", [GDNState, KDAState])
def test_native_accept_restore_and_reuse_preserve_other_requests(cls):
    owner, _, _ = make_state(cls, step=2, device="cpu")
    owner.conv_state.copy_(torch.arange(5))
    for slot in range(owner.ssm_state.shape[1]):
        owner.ssm_state[:, slot].fill_(slot)
    before = owner.ssm_state.clone()
    owner.commit_bounded_att_state(torch.tensor([1]), torch.tensor([3], dtype=torch.int32))
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
    monkeypatch.setattr(base, "linear_att_mtp_state_index_update", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(owner._ssm, "accept_updates", lambda *a: pytest.fail("unfused acceptance"))
    owner.commit_bounded_att_state(torch.tensor([2, 0, 1]), torch.tensor([1, 2, 3]), context=make_commit_context(()))
    assert len(calls) == 1
    assert calls[0]["replay_cursors"] is owner._ssm.cursors
    assert calls[0]["req_to_mtp_state_index"] is owner.mtp_state_index


@pytest.mark.parametrize("change", ["owner", "shape", "optional", "dtype"])
def test_graph_layout_mismatch_rejected_before_copy(change):
    from copy import copy

    owner, _, _ = make_state(mode="native", device="cpu")
    target = owner.prepare_decode(torch.tensor([0, 1], dtype=torch.int32), torch.tensor([0, 3, 6]))
    source = owner.prepare_decode(torch.tensor([1, 2], dtype=torch.int32), torch.tensor([0, 2, 5]))
    if change == "owner":
        source.resource = copy(owner)
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


def test_forward_contexts_select_each_layer_resource_and_publish_both_commits():
    import weakref
    from dataclasses import replace
    from lightllm.common.attention_state import AttentionStateManager
    from lightllm.common.basemodel.attention.linear.gdn import GDNLinearAttDecodeAttState
    from lightllm.common.basemodel.batch_objs import PostLayerOutput
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from lightllm.common.basemodel.hidden_collector import NoopHiddenCollector
    from lightllm.common.basemodel.infer_struct import InferStateInfo

    first, _, _ = make_state(mode="compact", device="cpu")
    second = GDNState(
        replace(first.config, logical_layer_ids=(3,), ssm_dtype=torch.bfloat16),
        4,
        mtp_step=2,
        device="cpu",
        ssm_state_mode="compact",
    )
    resources = AttentionStateManager(request_capacity=4)
    resources.register_state("main", first)
    resources.register_state("main", second)
    resources.register_state("draft0", GDNState(first.config, 4, mtp_step=2, device="cpu"))
    resources.finish_state_registration()
    infer = InferStateInfo()
    infer.batch_size = 6
    infer.b_req_idx = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32)
    infer.att_state_manager = resources
    infer.hidden_collector = NoopHiddenCollector()
    backend = SimpleNamespace(
        model=SimpleNamespace(is_mtp_draft_model=False, mtp_manager=SimpleNamespace(get_decode_draft_step=lambda _: 2)),
        uses_dynamic_spec_verify_layout=lambda: False,
    )
    att = GDNLinearAttDecodeAttState(backend=backend, _infer_state_weak_ref=weakref.ref(infer))
    infer.decode_att_state1 = att
    att.init_state()
    assert len(att.forwards) == 2
    assert att.get_forward(0) is att.get_forward(1)
    assert att.get_forward(3).resource is second
    configs = [SSMRunConfig(8, 1), SSMRunConfig(32, 2)]
    att.get_forward(0).run_config, att.get_forward(3).run_config = configs
    output = TpPartBaseModel._create_model_output(None, PostLayerOutput(torch.ones(6, 1)), infer)
    assert {c.state_name: c.payload for c in output.state_commits} == {
        first.state_name: configs[0],
        second.state_name: configs[1],
    }


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
