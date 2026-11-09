"""GDN replay 的推理入口。

Triton 和 FlashQla 各有一个 backend。它们只差 prefill chunk kernel，
decode 都走 ReplayGDNState。两个类都直接实现 backend 接口，不继承 native GDN。
"""

from __future__ import annotations

import dataclasses
import weakref
from typing import TYPE_CHECKING

import torch

from lightllm.common.attention_state.bounded_att.gdn.replay_gdn_state import (
    ReplayGDNState,
)
from lightllm.common.basemodel.attention.base_att import (
    AttControl,
    BaseAttBackend,
    BaseDecodeAttState,
    BasePrefillAttState,
)
from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d import (
    causal_conv1d_fn,
)
from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d_mtp import (
    causal_conv1d_update as causal_conv1d_update_mtp,
)
from lightllm.common.basemodel.triton_kernel.linear_att.fused_gdn_gating import (
    fused_gdn_gating,
)
from lightllm.common.basemodel.triton_kernel.linear_att.gdn_decode_pack import (
    conv_pack_gdn_decode_inputs,
)
from lightllm.common.basemodel.triton_kernel.linear_att.mtp_state_params import (
    build_dynamic_mtp_linear_att_state_params,
)
from lightllm.utils.envs_utils import get_env_start_args

from .replay_gdn_step import begin_decode, replay

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo


def replay_gdn_backend_class(name: str):
    """按已选中的 linear kernel 名字返回对应的 replay backend。"""

    classes = {
        "triton": ReplayGDNTritonAttBackend,
        "flashqla": ReplayGDNFlashQlaAttBackend,
    }
    try:
        return classes[name]
    except KeyError:
        raise ValueError(f"replay GDN has no backend named {name!r}") from None


def _triton_prefill_kernel():
    from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops import (
        chunk_gated_delta_rule,
    )

    return chunk_gated_delta_rule


def _flashqla_prefill_kernel():
    from flash_qla import chunk_gated_delta_rule

    return chunk_gated_delta_rule


def _configure_gdn(backend, model, prefill_kernel) -> None:
    """写入 GDN 的张量并行尺寸，并绑定这个 backend 自己的 prefill kernel。"""

    config = model.config
    tp_world_size = model.tp_world_size_
    backend.mtp_step = get_env_start_args().mtp_step
    backend.num_v_heads = config["linear_num_value_heads"]
    backend.num_k_heads = config["linear_num_key_heads"]
    backend.head_k_dim = config["linear_key_head_dim"]
    backend.head_v_dim = config["linear_value_head_dim"]
    backend.key_dim = backend.head_k_dim * backend.num_k_heads
    backend.value_dim = backend.head_v_dim * backend.num_v_heads
    backend.conv_kernel_dim = config["linear_conv_kernel_dim"]
    backend.activation = config["hidden_act"]
    backend.tp_qkvz_dim = (backend.key_dim * 2 + backend.value_dim * 2) // tp_world_size
    backend.tp_ba_dim = (backend.num_v_heads * 2) // tp_world_size
    backend.tp_num_k_heads = backend.num_k_heads // tp_world_size
    backend.tp_num_v_heads = backend.num_v_heads // tp_world_size
    backend.tp_key_dim = backend.key_dim // tp_world_size
    backend.tp_value_dim = backend.value_dim // tp_world_size
    assert backend.num_v_heads % backend.num_k_heads == 0, "num_v_heads must be divisible by num_k_heads"
    backend.num_v_heads_per_k_head = backend.num_v_heads // backend.num_k_heads
    ssm_dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    backend.ssm_state_dtype = ssm_dtypes.get(get_env_start_args().linear_att_ssm_data_type, torch.bfloat16)
    backend.prefill_kernel = prefill_kernel


def _split_qkvzba(backend, mixed_qkvzba):
    qkv_dim = backend.tp_key_dim * 2 + backend.tp_value_dim
    z_end = qkv_dim + backend.tp_value_dim
    b_end = z_end + backend.tp_num_v_heads
    mixed_qkv = mixed_qkvzba[:, :qkv_dim]
    z = mixed_qkvzba[:, qkv_dim:z_end].view(-1, backend.tp_num_v_heads, backend.head_v_dim)
    b = mixed_qkvzba[:, z_end:b_end]
    a = mixed_qkvzba[:, b_end:]
    return mixed_qkv, z, b, a


def _rearrange_mixed_qkv(backend, mixed_qkv, decode=False):
    query, key, value = torch.split(
        mixed_qkv,
        [backend.tp_key_dim, backend.tp_key_dim, backend.tp_value_dim],
        dim=-1,
    )
    if decode:
        batch_size = mixed_qkv.shape[0]
        leading = (batch_size, 1)
    else:
        leading = (1, query.shape[0])
    query = query.view(*leading, backend.tp_num_k_heads, backend.head_k_dim)
    key = key.view(*leading, backend.tp_num_k_heads, backend.head_k_dim)
    value = value.view(*leading, backend.tp_num_v_heads, backend.head_v_dim)
    return query, key, value


def _replay_state(att_state, layer_num: int) -> ReplayGDNState:
    return att_state.infer_state.att_state_manager.get_state(
        model_role=att_state.infer_state.att_state_role,
        state_type_filter="gdn",
        logical_layer_id=layer_num,
        state_class=ReplayGDNState,
    )


def _single_replay_state(att_state) -> ReplayGDNState:
    role = att_state.infer_state.att_state_role
    matches = [
        state
        for state in att_state.infer_state.att_state_manager.get_bounded_attention_states(role)
        if isinstance(state, ReplayGDNState)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one ReplayGDNState for {role}")
    return matches[0]


class ReplayGDNTritonAttBackend(BaseAttBackend):
    """GDN replay 的 Triton 入口。Prefill 使用仓内 chunk kernel。"""

    def __init__(self, model):
        super().__init__(model=model)
        _configure_gdn(self, model, _triton_prefill_kernel())

    def create_att_prefill_state(self, infer_state: "InferStateInfo"):
        return ReplayGDNPrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state: "InferStateInfo"):
        return ReplayGDNDecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


class ReplayGDNFlashQlaAttBackend(BaseAttBackend):
    """GDN replay 的 FlashQla 入口。Prefill 使用 flash_qla chunk kernel。"""

    def __init__(self, model):
        super().__init__(model=model)
        _configure_gdn(self, model, _flashqla_prefill_kernel())

    def create_att_prefill_state(self, infer_state: "InferStateInfo"):
        return ReplayGDNPrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state: "InferStateInfo"):
        return ReplayGDNDecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class ReplayGDNPrefillAttState(BasePrefillAttState):
    b_conv_buffer_idx: torch.Tensor = None
    b_ssm_buffer_idx: torch.Tensor = None

    def init_state(self):
        # Prefill 直接覆盖 canonical SSM。进入前 history 已经在 decode 接受时折完。
        reqs = self.infer_state.b_req_idx
        self.b_conv_buffer_idx = reqs
        self.b_ssm_buffer_idx = reqs

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        assert att_control.linear_att_prefill and att_control.linear_att_prefill_dict is not None
        params = att_control.linear_att_prefill_dict
        layer_num = params["layer_num"]
        layer_weight = params["layer_weight"]
        backend = self.backend
        state = _replay_state(self, layer_num)
        conv_states, ssm_states = state.get_layer_view(logical_layer_id=layer_num)
        if backend.mtp_step > 0:
            conv_states = conv_states[:, :, : -backend.mtp_step]
        mixed_qkv, z, b, a = _split_qkvzba(backend, params["mixed_qkvzba"])
        core_attn_out = self._prefill(mixed_qkv, conv_states, ssm_states, a, b, layer_weight)
        state.close_history_after_layer(layer_num, self.b_conv_buffer_idx, fold=False)
        return core_attn_out, z

    def _prefill(self, mixed_qkv, conv_states, ssm_states, a, b, layer_weight):
        backend = self.backend
        g, beta = fused_gdn_gating(layer_weight.linear_A_log.weight, a, b, layer_weight.linear_dt_bias.weight)
        mixed_qkv = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            layer_weight.linear_conv1d.mm_param.weight,
            bias=layer_weight.linear_conv1d.bias,
            query_start_loc=self.infer_state.b1_cu_q_seq_len,
            cache_indices=self.b_conv_buffer_idx,
            has_initial_state=self.infer_state.b_ready_cache_len > 0,
            conv_states=conv_states,
            activation=backend.activation,
        ).transpose(0, 1)
        query, key, value = _rearrange_mixed_qkv(backend, mixed_qkv)
        core_attn_out, last_state = backend.prefill_kernel(
            q=query,
            k=key,
            v=value,
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=ssm_states[self.b_ssm_buffer_idx],
            output_final_state=True,
            cu_seqlens=self.infer_state.b1_cu_q_seq_len,
            use_qk_l2norm_in_kernel=True,
        )
        ssm_states[self.b_ssm_buffer_idx] = last_state.to(ssm_states.dtype, copy=False)
        return core_attn_out


@dataclasses.dataclass
class ReplayGDNDecodeAttState(BaseDecodeAttState):
    b_conv_buffer_idx: torch.Tensor = None
    b_ssm_buffer_idx: torch.Tensor = None
    b1_mtp_cu_q_seq_len: torch.Tensor = None
    b_num_accepted_tokens: torch.Tensor = None

    def init_state(self):
        state = _single_replay_state(self)
        draft_step = self.backend.model.mtp_manager.get_decode_draft_step(self.backend.model.is_mtp_draft_model)
        self._init_indices(draft_step, state.mtp_state_index)
        self.b_ssm_buffer_idx = self.b_conv_buffer_idx
        begin_decode(state, self.b_conv_buffer_idx, self.b1_mtp_cu_q_seq_len)

    def _init_indices(self, draft_step: int, mtp_index: torch.Tensor | None) -> None:
        infer = self.infer_state
        if draft_step == 0:
            self.b_conv_buffer_idx = infer.b_req_idx
            self.b1_mtp_cu_q_seq_len = None
            self.b_num_accepted_tokens = None
            return
        if self.backend.uses_dynamic_spec_verify_layout():
            cu, conv_idx, accepted = build_dynamic_mtp_linear_att_state_params(
                b_req_idx=infer.b_req_idx,
                b_mtp_index=infer.b_mtp_index,
                req_to_mtp_state_index=mtp_index,
                hold_req_id=infer.req_manager.HOLD_REQUEST_ID,
            )
            self.b1_mtp_cu_q_seq_len = cu
            self.b_conv_buffer_idx = conv_idx
            self.b_num_accepted_tokens = accepted
            return
        width = draft_step + 1
        batch_size = infer.batch_size
        if batch_size % width:
            raise ValueError(
                "GDN replay decode requires batch_size divisible by draft_step + 1, "
                f"got batch_size={batch_size}, draft_step={draft_step}."
            )
        self.b1_mtp_cu_q_seq_len = torch.arange(
            0, batch_size + 1, width, dtype=torch.int32, device=infer.b_req_idx.device
        )
        self.b_conv_buffer_idx = infer.b_req_idx.view(-1, width)[:, 0].contiguous()
        self.b_num_accepted_tokens = mtp_index[self.b_conv_buffer_idx] + 1

    def decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        assert att_control.linear_att_decode and att_control.linear_att_decode_dict is not None
        params = att_control.linear_att_decode_dict
        layer_num = params["layer_num"]
        layer_weight = params["layer_weight"]
        backend = self.backend
        mixed_qkv, z, b, a = _split_qkvzba(backend, params["mixed_qkvzba"])
        state = _replay_state(self, layer_num)
        conv_states, _ = state.get_layer_view(logical_layer_id=layer_num)
        draft_step = backend.model.mtp_manager.get_decode_draft_step(backend.model.is_mtp_draft_model)
        if draft_step > 0:
            core = self._mtp(mixed_qkv, conv_states, a, b, layer_weight, layer_num, draft_step, state)
            return core, z
        return self._decode(mixed_qkv, z, conv_states, a, b, layer_weight, layer_num, state)

    def _decode(self, mixed_qkv, z, conv_states, a, b, layer_weight, layer_num, state):
        backend = self.backend
        query, key, value, z, a, b = conv_pack_gdn_decode_inputs(
            mixed_qkv,
            z,
            a,
            b,
            conv_states,
            layer_weight.linear_conv1d.mm_param.weight,
            layer_weight.linear_conv1d.bias,
            self.b_conv_buffer_idx,
            backend.activation,
            backend.conv_kernel_dim,
            backend.tp_num_k_heads,
            backend.head_k_dim,
            backend.tp_num_v_heads,
            backend.head_v_dim,
        )
        output, _ = replay(
            state,
            state.storage_layer(layer_num),
            query,
            key,
            value,
            a,
            b,
            layer_weight.linear_A_log.weight,
            layer_weight.linear_dt_bias.weight,
            self.b_conv_buffer_idx,
            self.b1_mtp_cu_q_seq_len,
        )
        state.close_history_after_layer(layer_num, self.b_conv_buffer_idx, fold=True)
        return output, z

    def _mtp(self, mixed_qkv, conv_states, a, b, layer_weight, layer_num, draft_step, state):
        backend = self.backend
        mixed_qkv = causal_conv1d_update_mtp(
            mixed_qkv,
            conv_states,
            layer_weight.linear_conv1d.mm_param.weight,
            mtp_step=draft_step,
            bias=layer_weight.linear_conv1d.bias,
            activation=backend.activation,
            conv_state_indices=self.b_conv_buffer_idx,
            num_accepted_tokens=self.b_num_accepted_tokens,
            query_start_loc=self.b1_mtp_cu_q_seq_len,
        )
        query, key, value = _rearrange_mixed_qkv(backend, mixed_qkv, decode=False)
        output, _ = replay(
            state,
            state.storage_layer(layer_num),
            query,
            key,
            value,
            a,
            b,
            layer_weight.linear_A_log.weight,
            layer_weight.linear_dt_bias.weight,
            self.b_conv_buffer_idx,
            self.b1_mtp_cu_q_seq_len,
        )
        return output
