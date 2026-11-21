"""GDN linear attention adapters over the gdn backend family.

Model layers pass the layer's operands, parameters and recurrent state views in
AttControl.linear_att_*_dict (see Qwen3NextTransformerLayerInfer._gdn_att_dict); the adapter builds the
layer config, runs the impl and, for replay state, advances the replay cursors after the last layer.
"""

import dataclasses
import weakref
from typing import TYPE_CHECKING, Optional

import torch

from lightllm.common.attention.backends.gdn import GDNLayerConfig, GDNLayerInputs
from lightllm.common.attention.backends.gdn.flashqla import FlashQLAGDNPrefillBackend
from lightllm.common.attention.backends.gdn.replay import ReplayGDNDecodeBackend
from lightllm.common.attention.backends.gdn.triton import (
    GDNDecodeStepArgs,
    GDNPrefillStepArgs,
    TritonGDNDecodeBackend,
    TritonGDNPrefillBackend,
)
from lightllm.utils.envs_utils import get_env_start_args
from ..base_att import AttControl, BaseAttBackend, BaseDecodeAttState, BasePrefillAttState, build_attention_batch

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from lightllm.common.basemodel.infer_struct import InferStateInfo


class LinearAttBackend(BaseAttBackend):
    """GDN over native recurrent state with the Triton chunked prefill; subclasses pick other variants."""

    prefill_backend = TritonGDNPrefillBackend
    decode_backend = TritonGDNDecodeBackend
    replay = False

    def __init__(self, model: "TpPartBaseModel"):
        super().__init__(model=model)
        config, tp_world_size = model.config, model.tp_world_size_
        self.mtp_step = get_env_start_args().mtp_step
        self.tp_num_k_heads = config["linear_num_key_heads"] // tp_world_size
        self.tp_num_v_heads = config["linear_num_value_heads"] // tp_world_size
        self.head_k_dim = config["linear_key_head_dim"]
        self.head_v_dim = config["linear_value_head_dim"]
        self.conv_kernel_dim = config["linear_conv_kernel_dim"]
        self.activation = config["hidden_act"]
        # Resolve the chunk kernel now, so a missing optional library fails at startup.
        self.prefill_backend.get_impl_cls().chunk_gated_delta_rule()
        self.prefill_step_builder = self.prefill_backend.get_step_builder_cls()(
            ssm_slots_per_request=1 if self.replay else self.mtp_step + 1
        )
        # BaseAttBackend reuses one instance per model but reruns __init__ on every construction.
        # Keep the decode step builder: captured CUDA graphs read the tensors it keeps.
        if getattr(self, "decode_step_builder", None) is None:
            self.decode_step_builder = self.decode_backend.get_step_builder_cls()(replay=self.replay)

    def layer(self, params: dict, dtype: torch.dtype):
        layer_config = GDNLayerConfig(
            num_k_heads=self.tp_num_k_heads,
            num_v_heads=self.tp_num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_kernel_dim=self.conv_kernel_dim,
            activation=self.activation,
            dtype=dtype,
            state_mtp_step=self.mtp_step,
        )
        layer_inputs = GDNLayerInputs(
            mixed_qkv=params["mixed_qkv"],
            a=params["a"],
            b=params["b"],
            z=params["z"],
            conv_weight=params["conv_weight"],
            conv_bias=params["conv_bias"],
            A_log=params["A_log"],
            dt_bias=params["dt_bias"],
            conv_states=params["conv_states"],
            ssm_states=params.get("ssm_states"),
            replay_state=params["state"] if self.replay else None,
            layer_num=params["layer_num"],
        )
        return layer_inputs, layer_config

    def create_att_prefill_state(self, infer_state: "InferStateInfo") -> "LinearAttPrefillAttState":
        return LinearAttPrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state: "InferStateInfo") -> "LinearAttDecodeAttState":
        return LinearAttDecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


def _run(backend: LinearAttBackend, backend_cls, step_args, params: dict, alloc_func):
    layer_inputs, layer_config = backend.layer(params, params["mixed_qkv"].dtype)
    impl = backend.get_attention_impl(backend_cls, layer_config)
    z = layer_inputs.z
    out = alloc_func((z.shape[0], layer_config.num_v_heads, layer_config.head_v_dim), dtype=z.dtype, device=z.device)
    return impl.forward(step_args, layer_inputs, out=out, alloc_func=alloc_func)


@dataclasses.dataclass
class LinearAttPrefillAttState(BasePrefillAttState):
    step_args: Optional[GDNPrefillStepArgs] = None

    def init_state(self):
        # The prefill CUDA graph's CPU callbacks enter through the runtime infer_state's state, so they
        # read the indices built here for the current batch (padding requests included).
        self.step_args = self.backend.prefill_step_builder.build(build_attention_batch(self.infer_state, self.backend))

    def prefill_att(self, q, k, v, att_control: AttControl = AttControl(), alloc_func=torch.empty) -> torch.Tensor:
        assert att_control.linear_att_prefill and att_control.linear_att_prefill_dict is not None
        params = att_control.linear_att_prefill_dict
        return _run(self.backend, self.backend.prefill_backend, self.step_args, params, alloc_func)


@dataclasses.dataclass
class LinearAttDecodeAttState(BaseDecodeAttState):
    step_args: Optional[GDNDecodeStepArgs] = None

    def init_state(self):
        batch = build_attention_batch(self.infer_state, self.backend)
        if batch.decode_rows_per_request > 1:
            # Speculative rows resume from each request's last accepted draft, kept by the recurrent state.
            batch = dataclasses.replace(batch, req_to_mtp_state_index=_recurrent_state(self).mtp_state_index)
        # Replay steps also build: the step builder rewrites the tensors the graph captured.
        self.step_args = self.backend.decode_step_builder.build(batch)

    def decode_att(self, q, k, v, att_control: AttControl = AttControl(), alloc_func=torch.empty) -> torch.Tensor:
        assert att_control.linear_att_decode and att_control.linear_att_decode_dict is not None
        params = att_control.linear_att_decode_dict
        out = _run(self.backend, self.backend.decode_backend, self.step_args, params, alloc_func)
        if self.backend.replay:
            # Every layer read the cursors of the step's start; the state advances them after the last layer.
            params["state"].finish_decode_token(params["layer_num"], self.step_args.conv_idx, self.step_args.cu_seqlens)
        return out


def _recurrent_state(att_state):
    """The role's single GDN state, which holds the speculative acceptance index."""
    from lightllm.common.attention_state.bounded_att import GDNState
    from lightllm.common.attention_state.bounded_att.gdn.replay_gdn_state import ReplayGDNState

    infer_state = att_state.infer_state
    state_class = ReplayGDNState if att_state.backend.replay else GDNState
    matches = [
        state
        for state in infer_state.att_state_manager.get_bounded_attention_states(infer_state.att_state_role)
        if isinstance(state, state_class)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {state_class.__name__} for {infer_state.att_state_role}")
    return matches[0]
