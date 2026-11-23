"""KDA backends at the step-builder/impl level against a sequential fp32 reference: native and replay
prefill (Triton or FlashKDA), then native and replay decode with and without speculative rows."""

import dataclasses

from functools import partial

import pytest
import torch

from lightllm.common.attention.backends.kda import KDALayerConfig, KDALayerInputs
from lightllm.common.attention.backends.kda.triton import (
    KDAPrefillBuildContext,
    ReplayKDADecodeBackend,
    ReplayKDAPrefillBackend,
    TritonKDADecodeBackend,
    TritonKDAPrefillBackend,
)
from unit_tests.common.attention.backends.linear_ref import CAPACITY, KDAShape, kda_token, run_prefill_then_decode

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _flash_kda_fwd():
    try:
        from flash_kda import fwd
    except Exception:  # a broken build raises more than ImportError
        return None
    if torch.cuda.get_device_capability() != (9, 0):
        return None
    return fwd


class KDAFamily:
    def __init__(self, mtp_step):
        self.shape = KDAShape()
        s = self.shape
        self.config = KDALayerConfig(s.h, s.d, s.width, s.lower_bound, torch.bfloat16, state_mtp_step=mtp_step)
        self.params = {
            "conv_weight": torch.randn(s.conv_channels, s.width, device="cuda").mul_(0.3).bfloat16(),
            "A_log": torch.randn(s.h, device="cuda").mul_(0.5),
            "dt_bias": torch.randn(s.h * s.d, device="cuda").mul_(0.5),
        }
        self.token_fn = partial(kda_token, s, self.params)

    def make_state(self, *, replay, mtp_step):
        from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig
        from lightllm.common.attention_state.bounded_att.kda.create_kda_state import create_kda_state

        s = self.shape
        config = ConvSSMStateConfig((0,), s.h, s.h, s.d, s.d, s.width - 1, torch.bfloat16, torch.float32)
        mode = "replay" if replay else "native"
        return create_kda_state(
            config,
            CAPACITY,
            mtp_step=mtp_step,
            device="cuda",
            ssm_state_mode=mode,
            gate_lower_bound=s.lower_bound,
        )

    def make_inputs(self, rows):
        s = self.shape
        bf16 = dict(device="cuda", dtype=torch.bfloat16)
        return {
            "mixed_qkv": torch.randn(rows, s.conv_channels, **bf16),
            "raw_gate": torch.randn(rows, s.h * s.d, **bf16),
            "raw_beta": torch.randn(rows, s.h, **bf16),
        }

    def token_args(self, tensors, start, end):
        return tensors["mixed_qkv"][start:end], tensors["raw_gate"][start:end], tensors["raw_beta"][start:end]

    def layer_inputs(self, tensors, conv, ssm, replay_state):
        return KDALayerInputs(
            **tensors,
            conv_weight=self.params["conv_weight"],
            A_log=self.params["A_log"],
            dt_bias=self.params["dt_bias"],
            conv_states=conv,
            ssm_states=ssm,
            replay_state=replay_state,
            layer_num=0,
        )

    def alloc_out(self, rows):
        return torch.empty(rows, self.shape.h, self.shape.d, device="cuda", dtype=torch.bfloat16)


PREFILL_KERNELS = [
    pytest.param("triton", id="triton-prefill"),
    pytest.param(
        "flashkda",
        id="flashkda-prefill",
        marks=pytest.mark.skipif(_flash_kda_fwd() is None, reason="FlashKDA not usable on this GPU"),
    ),
]


@pytest.mark.parametrize("prefill", PREFILL_KERNELS)
@pytest.mark.parametrize("replay", [False, True], ids=["native", "replay"])
@pytest.mark.parametrize("mtp_step,dynamic", [(0, False), (2, False), (2, True)], ids=["mtp0", "fixed", "dynamic"])
def test_kda_prefill_then_speculative_decode_matches_reference(prefill, replay, mtp_step, dynamic):
    torch.manual_seed(13)
    family = KDAFamily(mtp_step)
    family.config = dataclasses.replace(family.config, replay=replay)
    prefill_cls = ReplayKDAPrefillBackend if replay else TritonKDAPrefillBackend
    decode_cls = ReplayKDADecodeBackend if replay else TritonKDADecodeBackend
    run_prefill_then_decode(
        family,
        replay=replay,
        mtp_step=mtp_step,
        dynamic=dynamic,
        prefill_impl=prefill_cls(family.config),
        decode_impl=decode_cls(family.config),
        prefill_builder=KDAPrefillBuildContext(
            ssm_slots_per_request=1 if replay else mtp_step + 1,
            flash_kda_fwd=_flash_kda_fwd() if prefill == "flashkda" else None,
        ),
    )
