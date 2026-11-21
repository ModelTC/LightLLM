# SPDX-License-Identifier: Apache-2.0

"""KDA linear attention adapters (GLM-5-Next) over the kda backend family.

Model layers pass the layer's operands, parameters and recurrent state views in
AttControl.linear_att_*_dict; the adapter builds the layer config, runs the impl and, for replay
state, advances the replay cursors after the last layer.
"""

from __future__ import annotations

import dataclasses
import weakref
from typing import TYPE_CHECKING, Optional

import torch

from lightllm.common.attention.backends.gdn.triton import GDNDecodeStepArgs
from lightllm.common.attention.backends.kda import KDALayerConfig, KDALayerInputs
from lightllm.common.attention.backends.kda.triton import (
    KDAPrefillStepArgs,
    TritonKDADecodeBackend,
    TritonKDAPrefillBackend,
)
from lightllm.common.basemodel.attention.base_att import (
    AttControl,
    BaseAttBackend,
    BaseDecodeAttState,
    BasePrefillAttState,
    build_attention_batch,
)
from lightllm.common.basemodel.attention.linear.flashkda import get_flash_kda_fwd

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from lightllm.common.basemodel.infer_struct import InferStateInfo


class KDALinearAttBackend(BaseAttBackend):
    """KDA over native recurrent state; ReplayKDALinearAttBackend uses replay state."""

    prefill_backend = TritonKDAPrefillBackend
    decode_backend = TritonKDADecodeBackend
    replay = False

    def __init__(self, model: "TpPartBaseModel"):
        super().__init__(model=model)
        config = model.config["linear_attn_config"]
        self.num_heads = config["num_heads"]
        self.head_dim = config["head_dim"]
        assert self.num_heads % model.tp_world_size_ == 0
        self.tp_num_heads = self.num_heads // model.tp_world_size_
        self.tp_hidden_size = self.tp_num_heads * self.head_dim
        self.conv_kernel_size = config["short_conv_kernel_size"]
        self.lower_bound = config.get("gate_lower_bound", -5.0)
        self.mtp_step = model.args.mtp_step
        self.flash_kda_fwd = get_flash_kda_fwd(model)
        backends = model.args.llm_prefill_att_backend
        requested = backends[1] if len(backends) > 1 else "auto"
        # Automatic selection keeps FlashKDA to the step lengths it was validated on.
        max_flash_len = (2048 if self.tp_num_heads < 16 else 8192) if requested == "auto" else None
        self.prefill_step_builder = self.prefill_backend.get_step_builder_cls()(
            ssm_slots_per_request=1 if self.replay else self.mtp_step + 1,
            flash_kda_fwd=self.flash_kda_fwd,
            max_flash_len=max_flash_len,
        )
        # BaseAttBackend reuses one instance per model but reruns __init__ on every construction.
        # Keep the decode step builder: captured CUDA graphs read the tensors it keeps.
        if getattr(self, "decode_step_builder", None) is None:
            self.decode_step_builder = self.decode_backend.get_step_builder_cls()(replay=self.replay)

    def layer(self, params: dict):
        layer_config = KDALayerConfig(
            num_heads=self.tp_num_heads,
            head_dim=self.head_dim,
            conv_kernel_size=self.conv_kernel_size,
            lower_bound=self.lower_bound,
            dtype=params["mixed_qkv"].dtype,
            state_mtp_step=self.mtp_step,
        )
        layer_inputs = KDALayerInputs(
            mixed_qkv=params["mixed_qkv"],
            raw_gate=params["raw_gate"],
            raw_beta=params["raw_beta"],
            conv_weight=params["conv_weight"],
            A_log=params["A_log"],
            dt_bias=params["dt_bias"],
            conv_states=params["conv_states"],
            ssm_states=params.get("ssm_states"),
            replay_state=params["state"] if self.replay else None,
            layer_num=params["layer_num"],
        )
        return layer_inputs, layer_config

    def create_att_prefill_state(self, infer_state: "InferStateInfo"):
        return KDAPrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state: "InferStateInfo"):
        return KDADecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


def _run(backend: KDALinearAttBackend, backend_cls, step_args, params: dict, alloc_func):
    layer_inputs, layer_config = backend.layer(params)
    impl = backend.get_attention_impl(backend_cls, layer_config)
    x = layer_inputs.mixed_qkv
    rows = x.shape[0]
    out = alloc_func((rows, layer_config.num_heads, layer_config.head_dim), dtype=x.dtype, device=x.device)
    return impl.forward(step_args, layer_inputs, out=out, alloc_func=alloc_func)


@dataclasses.dataclass
class KDAPrefillAttState(BasePrefillAttState):
    step_args: Optional[KDAPrefillStepArgs] = None

    def init_state(self):
        self.step_args = self.backend.prefill_step_builder.build(build_attention_batch(self.infer_state, self.backend))

    def prefill_att(self, q, k, v, att_control: AttControl = AttControl(), alloc_func=torch.empty):
        assert att_control.linear_att_prefill
        return _run(
            self.backend, self.backend.prefill_backend, self.step_args, att_control.linear_att_prefill_dict, alloc_func
        )


@dataclasses.dataclass
class KDADecodeAttState(BaseDecodeAttState):
    step_args: Optional[GDNDecodeStepArgs] = None

    def init_state(self):
        backend = self.backend
        batch = build_attention_batch(self.infer_state, backend)
        # KDA groups decode rows by the configured speculative depth.
        rows = backend.mtp_step + 1
        batch = dataclasses.replace(
            batch,
            decode_rows_per_request=rows,
            dynamic_decode_rows=rows > 1 and backend.uses_dynamic_spec_verify_layout(),
            req_to_mtp_state_index=_recurrent_state(self).mtp_state_index if rows > 1 else None,
        )
        # Replay steps also build: the step builder rewrites the tensors the graph captured.
        self.step_args = backend.decode_step_builder.build(batch)

    def decode_att(self, q, k, v, att_control: AttControl = AttControl(), alloc_func=torch.empty):
        assert att_control.linear_att_decode
        params = att_control.linear_att_decode_dict
        out = _run(self.backend, self.backend.decode_backend, self.step_args, params, alloc_func)
        if self.backend.replay:
            # Every layer read the cursors of the step's start; the state advances them after the last layer.
            params["state"].finish_decode_token(params["layer_num"], self.step_args.conv_idx, self.step_args.cu_seqlens)
        return out


def _recurrent_state(att_state):
    """The role's single KDA state, which holds the speculative acceptance index."""
    from lightllm.common.attention_state.bounded_att import KDAState
    from lightllm.common.attention_state.bounded_att.kda.replay_kda_state import ReplayKDAState

    infer_state = att_state.infer_state
    state_class = ReplayKDAState if att_state.backend.replay else KDAState
    matches = [
        state
        for state in infer_state.att_state_manager.get_bounded_attention_states(infer_state.att_state_role)
        if isinstance(state, state_class)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {state_class.__name__} for {infer_state.att_state_role}")
    return matches[0]
