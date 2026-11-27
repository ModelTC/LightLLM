"""GDN backends against a sequential fp32 reference, and the decode backends' CUDA graph storage."""

from copy import copy
from dataclasses import replace
from functools import partial

import pytest
import torch

from lightllm.common.attention.backends.gdn import GDNLayerConfig, GDNLayerInputs
from lightllm.common.attention.backends.gdn.flashqla import FlashQLAGDNPrefillBackend
from lightllm.common.attention.backends.gdn.triton import (
    ReplayGDNDecodeBackend,
    TritonGDNDecodeBackend,
    TritonGDNPrefillBackend,
)
from lightllm.common.basemodel.infer_struct import GraphMode
from unit_tests.common.attention.backends.linear_ref import (
    CAPACITY,
    HOLD,
    REQS,
    GDNShape,
    decode_batch,
    gdn_token,
    run_prefill_then_decode,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _flashqla_available():
    try:
        import flash_qla  # noqa: F401
    except Exception:  # a broken build raises more than ImportError
        return False
    return True


class GDNFamily:
    def __init__(self, mtp_step, shape=None):
        self.shape = shape or GDNShape()
        s = self.shape
        self.config = GDNLayerConfig(
            s.hk, s.hv, s.dk, s.dv, s.width, "silu", torch.bfloat16, state_draft_capacity=mtp_step
        )
        self.params = {
            "conv_weight": torch.randn(s.conv_channels, s.width, device="cuda").mul_(0.3).bfloat16(),
            "A_log": torch.randn(s.hv, device="cuda").mul_(0.5),
            "dt_bias": torch.randn(s.hv, device="cuda").mul_(0.5),
        }
        self.token_fn = partial(gdn_token, s, self.params)

    def make_state(self, *, replay, mtp_step):
        from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig, create_gdn_state

        s = self.shape
        config = ConvSSMStateConfig((0,), s.hk, s.hv, s.dk, s.dv, s.width - 1, torch.bfloat16, torch.float32)
        mode = "replay" if replay else "native"
        return create_gdn_state(config, CAPACITY, mtp_step=mtp_step, device="cuda", ssm_state_mode=mode)

    def make_inputs(self, rows):
        s = self.shape
        bf16 = dict(device="cuda", dtype=torch.bfloat16)
        return {
            "mixed_qkv": torch.randn(rows, s.conv_channels, **bf16),
            "a": torch.randn(rows, s.hv, **bf16),
            "b": torch.randn(rows, s.hv, **bf16),
            "z": torch.randn(rows, s.hv, s.dv, **bf16),
        }

    def token_args(self, tensors, start, end):
        return tensors["mixed_qkv"][start:end], tensors["a"][start:end], tensors["b"][start:end]

    def layer_inputs(self, tensors, conv, ssm, replay_state):
        return GDNLayerInputs(
            **tensors,
            conv_weight=self.params["conv_weight"],
            conv_bias=None,
            A_log=self.params["A_log"],
            dt_bias=self.params["dt_bias"],
            conv_states=conv,
            ssm_states=ssm,
            replay_state=replay_state,
            logical_layer_id=0,
        )

    def alloc_out(self, rows):
        return torch.empty(rows, self.shape.hv, self.shape.dv, device="cuda", dtype=torch.bfloat16)


PREFILL_IMPLS = [
    pytest.param("triton", id="triton-prefill"),
    pytest.param(
        "flashqla",
        id="flashqla-prefill",
        marks=pytest.mark.skipif(not _flashqla_available(), reason="flash_qla not usable"),
    ),
]


@pytest.mark.parametrize("prefill", PREFILL_IMPLS)
@pytest.mark.parametrize("replay", [False, True], ids=["native", "replay"])
@pytest.mark.parametrize("mtp_step,dynamic", [(0, False), (2, False), (2, True)], ids=["mtp0", "fixed", "dynamic"])
def test_gdn_prefill_then_speculative_decode_matches_reference(prefill, replay, mtp_step, dynamic):
    torch.manual_seed(11)
    # FlashQLA's chunk kernel supports only 128-dim key heads, as in Qwen3-Next.
    family = GDNFamily(mtp_step, GDNShape(dk=128, dv=128) if prefill == "flashqla" else None)
    family.config = replace(family.config, use_ssm_replay=replay)
    prefill_cls = FlashQLAGDNPrefillBackend if prefill == "flashqla" else TritonGDNPrefillBackend
    decode_cls = ReplayGDNDecodeBackend if replay else TritonGDNDecodeBackend
    run_prefill_then_decode(
        family,
        replay=replay,
        mtp_step=mtp_step,
        dynamic=dynamic,
        prefill_backend=prefill_cls(family.config),
        decode_backend=decode_cls(family.config),
    )


def _graph_batch(reqs, accepted_index, *, dynamic, microbatch_index, lengths=None):
    width = 3
    index = torch.zeros(CAPACITY, dtype=torch.int32, device="cuda")
    index[torch.tensor(reqs, device="cuda")] = torch.tensor(accepted_index, dtype=torch.int32, device="cuda")
    return decode_batch(
        reqs,
        lengths if dynamic else [width] * len(reqs),
        width,
        HOLD,
        index,
        dynamic=dynamic,
        microbatch_index=microbatch_index,
        graph=True,
    )


FIELDS = ("conv_idx", "ssm_idx", "cu_seqlens", "num_accepted_tokens")


@pytest.mark.parametrize("replay", [False, True], ids=["native", "replay"])
@pytest.mark.parametrize("dynamic", [False, True], ids=["fixed", "dynamic"])
def test_decode_backend_rewrites_graph_storage_in_place_per_microbatch(replay, dynamic):
    backend_cls = ReplayGDNDecodeBackend if replay else TritonGDNDecodeBackend
    config = GDNLayerConfig(2, 2, 16, 16, 4, "silu", torch.bfloat16, state_draft_capacity=2, use_ssm_replay=replay)
    backend = backend_cls(config)
    first = backend.build_att_metadata(
        _graph_batch([2, 0, 3], [1, 2, 0], dynamic=dynamic, microbatch_index=0, lengths=[3, 1, 2])
    )
    captured = {name: getattr(first, name) for name in FIELDS}
    ptrs = {name: t.data_ptr() for name, t in captured.items()}

    # A replay of the same graph: same microbatch and batch size, different requests and acceptance.
    batch = _graph_batch([1, 3, 2], [2, 0, 1], dynamic=dynamic, microbatch_index=0, lengths=[2, 3, 1])
    replayed = backend.build_att_metadata(batch)
    eager_batch = copy(batch)
    eager_batch.graph_mode = GraphMode.EAGER
    eager_batch.cuda_graph_max_kv_seq_len = None
    eager = backend_cls(config).build_att_metadata(eager_batch)
    for name in FIELDS:
        kept = getattr(replayed, name)
        assert kept.data_ptr() == ptrs[name], f"{name} moved between graph steps"
        assert torch.equal(captured[name], getattr(eager, name)), f"{name} kept the captured step's values"
    if replay:
        assert replayed.ssm_idx.data_ptr() == replayed.conv_idx.data_ptr()

    # The other microbatch of an overlapped step has its own storage, and building it leaves microbatch 0 alone.
    before = {name: getattr(replayed, name).clone() for name in FIELDS}
    other = backend.build_att_metadata(
        _graph_batch([0, 2, 3], [0, 0, 2], dynamic=dynamic, microbatch_index=1, lengths=[1, 2, 3])
    )
    for name in FIELDS:
        assert getattr(other, name).data_ptr() != ptrs[name], f"{name} shared across microbatches"
        assert torch.equal(getattr(replayed, name), before[name]), f"{name} of microbatch 0 overwritten"


def test_replay_decode_without_speculation_keeps_its_row_offsets():
    config = GDNLayerConfig(2, 2, 16, 16, 4, "silu", torch.bfloat16, use_ssm_replay=True)
    backend = ReplayGDNDecodeBackend(config)
    batch = decode_batch([2, 0, 3], [1, 1, 1], 1, HOLD, None, dynamic=False, graph=True)
    first = backend.build_att_metadata(batch)
    again = backend.build_att_metadata(decode_batch([1, 3, 2], [1, 1, 1], 1, HOLD, None, dynamic=False, graph=True))
    assert again.cu_seqlens.data_ptr() == first.cu_seqlens.data_ptr()
    assert torch.equal(again.cu_seqlens, torch.arange(4, dtype=torch.int32, device="cuda"))
    # Native state needs no derived tensors: the graph reads b_req_idx, which the caller rewrites in place.
    native = TritonGDNDecodeBackend(replace(config, use_ssm_replay=False)).build_att_metadata(batch)
    assert native.conv_idx is batch.b_req_idx and native.cu_seqlens is None


def test_reqs_avoid_the_hold_slot():
    assert HOLD not in REQS
