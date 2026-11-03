from lightllm.common.attention_state.bounded_att import GDNState
from .state import LinearDecodeAttState, LinearPrefillAttState, get_resource
import dataclasses
import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from ..base_att import BaseAttBackend, AttControl
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d import causal_conv1d_fn
from lightllm.common.basemodel.triton_kernel.linear_att.fused_gdn_gating import fused_gdn_gating
from lightllm.common.basemodel.triton_kernel.linear_att.gdn_decode_pack import conv_pack_gdn_decode_inputs

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from lightllm.common.basemodel.infer_struct import InferStateInfo
    from lightllm.models.qwen3next.infer_struct import Qwen3NextInferStateInfo
    from lightllm.models.qwen3next.layer_infer.transformer_layer_infer import Qwen3NextTransformerLayerWeight


class GDNLinearAttBackend(BaseAttBackend, ABC):
    def __init__(self, model: "TpPartBaseModel"):
        super().__init__(model=model)
        self._init_linear_layer_metadata(network_config=model.config, tp_world_size=model.tp_world_size_)
        self.prefill_kernel = self.get_prefill_kernel()

    @abstractmethod
    def get_prefill_kernel(self):
        pass

    def _init_linear_layer_metadata(self, network_config, tp_world_size):

        self.mtp_step = get_env_start_args().mtp_step

        # Linear attention specific dimensions
        self.num_v_heads = network_config["linear_num_value_heads"]
        self.num_k_heads = network_config["linear_num_key_heads"]
        self.head_k_dim = network_config["linear_key_head_dim"]
        self.head_v_dim = network_config["linear_value_head_dim"]
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_dim = network_config["linear_conv_kernel_dim"]
        self.activation = network_config["hidden_act"]

        # Tensor parallelism dimensions
        self.tp_qkvz_dim = (self.key_dim * 2 + self.value_dim * 2) // tp_world_size
        self.tp_ba_dim = (self.num_v_heads * 2) // tp_world_size
        self.tp_num_k_heads = self.num_k_heads // tp_world_size
        self.tp_num_v_heads = self.num_v_heads // tp_world_size
        self.tp_key_dim = self.key_dim // tp_world_size
        self.tp_value_dim = self.value_dim // tp_world_size

        assert self.num_v_heads % self.num_k_heads == 0, "num_v_heads must be divisible by num_k_heads"
        self.num_v_heads_per_k_head = self.num_v_heads // self.num_k_heads

        # SSM state dtype optimization
        ssm_dtype_dict = {"bfloat16": torch.bfloat16, "float32": torch.float32}
        self.ssm_state_dtype = ssm_dtype_dict.get(get_env_start_args().linear_att_ssm_data_type, torch.bfloat16)
        return

    def _split_qkvzba(self, mixed_qkvzba):
        qkv_dim = self.tp_key_dim * 2 + self.tp_value_dim
        z_end = qkv_dim + self.tp_value_dim
        b_end = z_end + self.tp_num_v_heads
        mixed_qkv = mixed_qkvzba[:, :qkv_dim]
        z = mixed_qkvzba[:, qkv_dim:z_end].view(-1, self.tp_num_v_heads, self.head_v_dim)
        b = mixed_qkvzba[:, z_end:b_end]
        a = mixed_qkvzba[:, b_end:]
        return mixed_qkv, z, b, a

    def _rearrange_mixed_qkv(self, mixed_qkv, decode=False):
        if decode:
            query, key, value = torch.split(
                mixed_qkv,
                [self.tp_key_dim, self.tp_key_dim, self.tp_value_dim],
                dim=-1,
            )
            batch_size = mixed_qkv.shape[0]
            query = query.view(batch_size, 1, self.tp_num_k_heads, self.head_k_dim)
            key = key.view(batch_size, 1, self.tp_num_k_heads, self.head_k_dim)
            value = value.view(batch_size, 1, self.tp_num_v_heads, self.head_v_dim)
            return query, key, value
        else:
            query, key, value = torch.split(
                mixed_qkv,
                [self.tp_key_dim, self.tp_key_dim, self.tp_value_dim],
                dim=-1,
            )
            seq_len = query.shape[0]
            query = query.view(1, seq_len, self.tp_num_k_heads, self.head_k_dim)
            key = key.view(1, seq_len, self.tp_num_k_heads, self.head_k_dim)
            value = value.view(1, seq_len, self.tp_num_v_heads, self.head_v_dim)
            return query, key, value

    def create_att_prefill_state(self, infer_state: "InferStateInfo") -> "GDNLinearAttPrefillAttState":
        return GDNLinearAttPrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state: "InferStateInfo") -> "GDNLinearAttDecodeAttState":
        return GDNLinearAttDecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class GDNLinearAttPrefillAttState(LinearPrefillAttState):
    state_class = GDNState

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert att_control.linear_att_prefill, "linear_att_prefill must be True for Linear prefill attention"
        assert att_control.linear_att_prefill_dict is not None, "linear_att_prefill_dict is required"

        linear_att_dict = att_control.linear_att_prefill_dict
        mixed_qkvzba: torch.Tensor = linear_att_dict["mixed_qkvzba"]
        layer_weight = linear_att_dict["layer_weight"]
        layer_num = linear_att_dict["layer_num"]
        backend: GDNLinearAttBackend = self.backend

        resource = get_resource(self, layer_num)
        conv_states, ssm_states = resource.get_layer_view(layer_num)
        # 在开启了mtp的时候，conv 状态的最后一维可能存在冗余的部分，需要进行切片对齐。
        # prefill 模式下，使用不到这几个维度，所以需要扣除掉，
        if backend.mtp_step > 0:
            conv_states = conv_states[:, :, : -backend.mtp_step]
        mixed_qkv, z, b, a = backend._split_qkvzba(mixed_qkvzba)
        core_attn_out = self._gdn_prefill_kernel(
            mixed_qkv,
            conv_states,
            ssm_states,
            a,
            b,
            self.infer_state,
            layer_weight,
            self.state_indices[resource.state_name],
        )
        return core_attn_out, z

    def _gdn_prefill_kernel(
        self,
        mixed_qkv: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        infer_state: "Qwen3NextInferStateInfo",
        layer_weight: "Qwen3NextTransformerLayerWeight",
        state_indices,
    ):
        g, beta = fused_gdn_gating(layer_weight.linear_A_log.weight, a, b, layer_weight.linear_dt_bias.weight)
        mixed_qkv = mixed_qkv.transpose(0, 1)

        conv_indices, ssm_indices = state_indices
        backend: GDNLinearAttBackend = self.backend
        out_tensor = causal_conv1d_fn(
            mixed_qkv,
            layer_weight.linear_conv1d.mm_param.weight,
            bias=layer_weight.linear_conv1d.bias,
            query_start_loc=infer_state.b1_cu_q_seq_len,
            cache_indices=conv_indices,
            has_initial_state=infer_state.b_ready_cache_len > 0,
            conv_states=conv_states,
            activation=backend.activation,
        )
        mixed_qkv = out_tensor.transpose(0, 1)

        # Recurrent processing
        query, key, value = backend._rearrange_mixed_qkv(mixed_qkv)
        initial_state = ssm_states[ssm_indices]
        # g and beta have shape (total_tokens, num_heads), need to unsqueeze to get (1, total_tokens, num_heads)
        core_attn_out, last_recurrent_state = backend.prefill_kernel(
            q=query,
            k=key,
            v=value,
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=infer_state.b1_cu_q_seq_len,
            use_qk_l2norm_in_kernel=True,
        )
        # The chunk kernel accumulates the recurrent state in float32 even when
        # the state cache is configured as bfloat16. Advanced indexing does
        # not perform an implicit dtype conversion for index_put.
        ssm_states[ssm_indices] = last_recurrent_state.to(ssm_states.dtype, copy=False)
        return core_attn_out


@dataclasses.dataclass
class GDNLinearAttDecodeAttState(LinearDecodeAttState):
    state_class = GDNState

    def get_draft_step(self):
        return self.backend.model.mtp_manager.get_decode_draft_step(self.backend.model.is_mtp_draft_model)

    def decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert att_control.linear_att_decode, "linear_att_decode must be True for Linear decode attention"
        assert att_control.linear_att_decode_dict is not None, "linear_att_decode_dict is required"

        linear_att_dict = att_control.linear_att_decode_dict
        mixed_qkvzba = linear_att_dict["mixed_qkvzba"]
        layer_weight = linear_att_dict["layer_weight"]
        layer_num = linear_att_dict["layer_num"]
        backend: GDNLinearAttBackend = self.backend

        mixed_qkv, z, b, a = backend._split_qkvzba(mixed_qkvzba)
        forward = self.get_forward(layer_num)
        conv_states, _ = forward.resource.get_layer_view(layer_num)

        draft_step = self.backend.model.mtp_manager.get_decode_draft_step(self.backend.model.is_mtp_draft_model)
        if draft_step > 0:
            core_attn_out = self._gdn_mtp_kernel(
                mixed_qkv,
                conv_states,
                forward,
                a,
                b,
                self.infer_state,
                layer_weight,
                layer_num=layer_num,
            )
        else:
            # 非 MTP 模式下，使用线性层 decode 状态。
            core_attn_out, z = self._gdn_decode_kernel(
                mixed_qkv,
                z,
                conv_states,
                forward,
                a,
                b,
                self.infer_state,
                layer_weight,
                layer_num=layer_num,
            )
        return core_attn_out, z

    def _gdn_decode_kernel(
        self,
        mixed_qkv: torch.Tensor,
        z: torch.Tensor,
        conv_states: torch.Tensor,
        forward,
        a: torch.Tensor,
        b: torch.Tensor,
        infer_state: "Qwen3NextInferStateInfo",
        layer_weight: "Qwen3NextTransformerLayerWeight",
        *,
        layer_num: int,
    ):
        backend: GDNLinearAttBackend = self.backend

        # Recurrent processing with fused gating. Decode uses a specialized
        # conv+pack kernel to avoid materializing the post-conv qkv tensor
        # before immediately splitting it into q/k/v.
        query, key, value, z, a, b = conv_pack_gdn_decode_inputs(
            mixed_qkv,
            z,
            a,
            b,
            conv_states,
            layer_weight.linear_conv1d.mm_param.weight,
            layer_weight.linear_conv1d.bias,
            forward.req_indices,
            backend.activation,
            backend.conv_kernel_dim,
            backend.tp_num_k_heads,
            backend.head_k_dim,
            backend.tp_num_v_heads,
            backend.head_v_dim,
        )
        core_attn_out = forward.run_layer(
            layer_num,
            query,
            key,
            value,
            a,
            b,
            layer_weight.linear_A_log.weight,
            layer_weight.linear_dt_bias.weight,
        )
        return core_attn_out, z

    def _gdn_mtp_kernel(
        self,
        mixed_qkv: torch.Tensor,
        conv_states: torch.Tensor,
        forward,
        a: torch.Tensor,
        b: torch.Tensor,
        infer_state: "Qwen3NextInferStateInfo",
        layer_weight: "Qwen3NextTransformerLayerWeight",
        *,
        layer_num: int,
    ):
        from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d_mtp import (
            causal_conv1d_update as causal_conv1d_update_mtp,
        )

        backend: GDNLinearAttBackend = self.backend

        cu_seqlens_q = forward.cu_seqlens
        mixed_qkv = causal_conv1d_update_mtp(
            mixed_qkv,
            conv_states,
            layer_weight.linear_conv1d.mm_param.weight,
            mtp_step=backend.model.mtp_manager.get_decode_draft_step(backend.model.is_mtp_draft_model),
            bias=layer_weight.linear_conv1d.bias,
            activation=backend.activation,
            conv_state_indices=forward.req_indices,
            num_accepted_tokens=forward.accepted_token_counts,
            query_start_loc=cu_seqlens_q,
        )

        query, key, value = backend._rearrange_mixed_qkv(mixed_qkv, decode=False)
        core_attn_out = forward.run_layer(
            layer_num,
            query,
            key,
            value,
            a,
            b,
            layer_weight.linear_A_log.weight,
            layer_weight.linear_dt_bias.weight,
        )
        return core_attn_out
