import pytest
import torch

from lightllm.common.attention_state.bounded_att import KDAState
from unit_tests.common.test_linear_state import make_state
from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig, GDNState
from lightllm.common.basemodel.triton_kernel.linear_att.ssm_autotune import resolve_run_config
from lightllm.common.attention_state.bounded_att.gdn.replay_gdn_state import ReplayGDNState
from lightllm.common.attention_state.bounded_att.kda.replay_kda_state import ReplayKDAState
from lightllm.models.glm5_next.config import Glm5NextResourceConfig
from lightllm.server.core.objs.start_args_type import StartArgs


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
    owner = ReplayKDAState(linear_config, 3, mtp_step=2, gate_lower_bound=-2.5, device="cpu")
    assert owner.gate_lower_bound == -2.5


def test_replay_backend_publishes_detached_run_config():
    ssm_backend = object.__new__(ReplayGDNState)
    tuning = {"BV": 32, "num_warps": 2}
    ssm_backend.run_config = tuning
    ssm_backend._config_is_fixed = True
    ssm_backend.compute_mode = "inline"
    config = resolve_run_config(ssm_backend, "replay", *([None] * 7), None)
    tuning["BV"] = 128
    assert config == {"BV": 32, "num_warps": 2, "num_stages": 3}


@pytest.mark.parametrize("cls", [GDNState, KDAState])
@pytest.mark.parametrize("mtp_step,expected", [(0, "inline"), (2, "precompute")])
def test_replay_defaults_to_precompute_for_speculative_decode(linear_config, cls, mtp_step, expected):
    replay_cls = ReplayKDAState if cls is KDAState else ReplayGDNState
    owner = replay_cls(linear_config, request_capacity=2, mtp_step=mtp_step, device="cpu")
    history = getattr(owner, "runtime", owner)
    assert history.compute_mode == expected
    assert history.verify_width == mtp_step + 1


@pytest.mark.parametrize("cls", [GDNState, KDAState])
def test_mtp0_decode_keeps_explicit_precompute(linear_config, cls):
    replay_cls = ReplayKDAState if cls is KDAState else ReplayGDNState
    owner = replay_cls(
        linear_config,
        request_capacity=2,
        mtp_step=0,
        device="cpu",
        ssm_replay_config={"cache_len": 8, "compute_mode": "precompute"},
    )
    assert owner.verify_width == 1
    assert owner.compute_mode == "precompute"
    assert owner.capacity == 8


class _CaptureKernel:
    def __init__(self):
        self.args = None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.args = args

        return launch


@pytest.mark.parametrize(
    "module_name,cls,bv,precompute",
    [
        ("replay_gdn_decode", ReplayGDNState, 32, True),
        ("replay_gdn_decode", ReplayGDNState, 8, False),
        ("replay_kda_decode", ReplayKDAState, 32, True),
        ("replay_kda_decode", ReplayKDAState, 8, False),
    ],
)
def test_mtp0_precompute_reaches_replay_kernel(monkeypatch, linear_config, module_name, cls, bv, precompute):
    import importlib

    decode = importlib.import_module(f"lightllm.common.basemodel.attention.linear.{module_name}")
    captured = _CaptureKernel()
    monkeypatch.setattr(decode, "_decode_replay", captured)
    owner = cls(
        linear_config,
        request_capacity=2,
        mtp_step=0,
        device="cpu",
        ssm_replay_config={"cache_len": 8, "compute_mode": "precompute"},
    )
    heads = owner.num_key_heads
    q = torch.zeros(1, 1, heads, linear_config.key_dim)
    v = torch.zeros(1, 1, owner.ssm_state.shape[-3], linear_config.value_dim)
    gate = v.shape[-3] * linear_config.key_dim if cls is ReplayKDAState else v.shape[-3]
    decode.replay(
        owner,
        0,
        q,
        q,
        v,
        torch.zeros(1, gate),
        torch.zeros(1, v.shape[-3]),
        torch.zeros(v.shape[-3]),
        torch.zeros(gate),
        torch.zeros(1, dtype=torch.int32),
        None,
        run_config={"BV": bv, "num_warps": 1, "num_stages": 1},
    )
    assert captured.args[33] is precompute
    assert captured.args[35] is (cls is ReplayKDAState)


@pytest.mark.parametrize("cls", [GDNState, KDAState])
def test_native_commit_selects_mtp_slot(cls):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    owner, _, _ = make_state(cls, step=2)
    owner.commit_bounded_att_state(
        torch.tensor([1], device="cuda"), torch.tensor([3], dtype=torch.int32, device="cuda")
    )
    assert owner.mtp_state_index.tolist() == [0, 2, 0, 0]


@pytest.mark.parametrize("cls", [GDNState, KDAState])
def test_replay_fused_acceptance_keeps_cursors_inside_mode(monkeypatch, cls):
    import sys

    owner, _, _ = make_state(cls, mode="replay", step=3, device="cpu")
    history = getattr(owner, "runtime", owner)
    calls = []
    monkeypatch.setattr(sys.modules[type(history).__module__], "commit_ssm_prefix", lambda *args: calls.append(args))
    monkeypatch.setattr(history, "accept_updates", lambda *a: pytest.fail("unfused acceptance"))
    owner.commit_bounded_att_state(torch.tensor([2, 0, 1]), torch.tensor([1, 2, 3]))
    assert len(calls) == 1
    assert calls[0][3] is history.cursors
    assert calls[0][0] is owner.mtp_state_index


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_acceptance_keeps_request_order_with_strided_inputs(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    owner, _, _ = make_state(mode="replay", step=3, device=device)
    owner.mtp_state_index.fill_(2)
    history = getattr(owner, "runtime", owner)
    history.cursors.fill_(8)
    reqs = torch.tensor([2, -99, 0, -99, 3, -99], device=device)[::2]
    lengths = torch.tensor([1, -99, 3, -99, 2, -99], dtype=torch.int32, device=device)[::2]
    owner.commit_accepted(owner.mtp_state_index, reqs, lengths)
    assert owner.mtp_state_index.tolist() == [2, 2, 0, 1]
    assert history.cursors.tolist() == [11, 8, 9, 8]
