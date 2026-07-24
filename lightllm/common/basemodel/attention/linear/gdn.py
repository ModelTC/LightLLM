import dataclasses
import torch
from typing import TYPE_CHECKING
from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from lightllm.utils.envs_utils import get_env_start_args, get_llm_data_type
from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d import causal_conv1d_fn
from lightllm.common.basemodel.triton_kernel.linear_att.fused_gdn_gating import fused_gdn_gating
from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops import chunk_gated_delta_rule
from lightllm.common.basemodel.triton_kernel.linear_att.gdn_decode_pack import conv_pack_gdn_decode_inputs
from lightllm.common.basemodel.triton_kernel.linear_att.mtp_fused_recurrent import (
    mtp_fused_recurrent_gated_delta_rule,
)
from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops import fused_recurrent_gated_delta_rule

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from lightllm.common.basemodel.infer_struct import InferStateInfo
    from lightllm.models.qwen3next.infer_struct import Qwen3NextInferStateInfo
    from lightllm.models.qwen3next.layer_infer.transformer_layer_infer import Qwen3NextTransformerLayerWeight


class LinearAttBackend(BaseAttBackend):
    def __init__(self, model: "TpPartBaseModel"):
        super().__init__(model=model)
        self._init_linear_layer_metadata(network_config=model.config, tp_world_size=model.tp_world_size_)

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
        start_args = get_env_start_args()
        self.ssm_state_dtype = ssm_dtype_dict.get(start_args.linear_att_ssm_data_type, torch.bfloat16)

        # Pre-compute whether dtype conversion is needed
        # GDN kernel output dtype is self.data_type
        # Conversion needed only if SSM state uses different dtype
        self.needs_ssm_dtype_conversion = get_llm_data_type() != self.ssm_state_dtype
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

    def create_att_prefill_state(self, infer_state: "InferStateInfo") -> "LinearAttPrefillAttState":
        return LinearAttPrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state: "InferStateInfo") -> "LinearAttDecodeAttState":
        return LinearAttDecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class LinearAttPrefillAttState(BasePrefillAttState):

    b_conv_buffer_idx: torch.Tensor = None
    b_ssm_buffer_idx: torch.Tensor = None

    def init_state(self):
        backend: LinearAttBackend = self.backend
        mtp_step = backend.mtp_step
        # 每次 _prefill 都会在 runtime infer_state 上调用 init_state。
        # prefill cuda graph 回调必须走 new_infer_state.prefill_att_state1，
        # 才能读到这里按当前 batch（含 token padding 后的 dummy request）更新的索引。
        self.b_conv_buffer_idx = self.infer_state.b_req_idx
        self.b_ssm_buffer_idx = self.infer_state.b_req_idx * (mtp_step + 1)
        return

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
        backend: LinearAttBackend = self.backend

        conv_states, ssm_states = self.infer_state.req_manager.get_mamba_cache(layer_num)
        # 在开启了mtp的时候，conv 状态的最后一维可能存在冗余的部分，需要进行切片对齐。
        # prefill 模式下，使用不到这几个维度，所以需要扣除掉，
        if backend.mtp_step > 0:
            conv_states = conv_states[:, :, : -backend.mtp_step]
        mixed_qkv, z, b, a = backend._split_qkvzba(mixed_qkvzba)
        core_attn_out = self._gdn_prefill_kernel(
            mixed_qkv, conv_states, ssm_states, a, b, self.infer_state, layer_weight
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
    ):
        g, beta = fused_gdn_gating(layer_weight.linear_A_log.weight, a, b, layer_weight.linear_dt_bias.weight)
        mixed_qkv = mixed_qkv.transpose(0, 1)

        backend: LinearAttBackend = self.backend
        out_tensor = causal_conv1d_fn(
            mixed_qkv,
            layer_weight.linear_conv1d.mm_param.weight,
            bias=layer_weight.linear_conv1d.bias,
            query_start_loc=infer_state.b1_cu_q_seq_len,
            cache_indices=self.b_conv_buffer_idx,
            has_initial_state=infer_state.b_ready_cache_len > 0,
            conv_states=conv_states,
            activation=backend.activation,
        )
        mixed_qkv = out_tensor.transpose(0, 1)

        # Recurrent processing
        query, key, value = backend._rearrange_mixed_qkv(mixed_qkv)
        initial_state = ssm_states[self.b_ssm_buffer_idx]
        # g and beta have shape (total_tokens, num_heads), need to unsqueeze to get (1, total_tokens, num_heads)
        core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=infer_state.b1_cu_q_seq_len,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        if backend.needs_ssm_dtype_conversion:
            ssm_states[self.b_ssm_buffer_idx] = last_recurrent_state.to(backend.ssm_state_dtype, copy=False)
        else:
            ssm_states[self.b_ssm_buffer_idx] = last_recurrent_state
        return core_attn_out


@dataclasses.dataclass
class LinearAttDecodeAttState(BaseDecodeAttState):

    b_conv_buffer_idx: torch.Tensor = None
    b_ssm_buffer_idx: torch.Tensor = None
    b1_mtp_cu_q_seq_len: torch.Tensor = None
    b_num_accepted_tokens: torch.Tensor = None

    def init_state(self):
        backend: LinearAttBackend = self.backend
        mtp_step = backend.mtp_step

        # decode 模式下
        if mtp_step == 0:
            # 非mtp模式下，不需要额外状态
            self.b_conv_buffer_idx = self.infer_state.b_req_idx
            self.b_ssm_buffer_idx = self.infer_state.b_req_idx
            return

        if mtp_step > 0:
            # mtp 模式下
            batch_size = self.infer_state.batch_size
            att_batch_size = batch_size // (mtp_step + 1)
            assert batch_size % (mtp_step + 1) == 0

            device = self.infer_state.b_req_idx.device

            # shape 为 [att_batch_size + 1]
            self.b1_mtp_cu_q_seq_len = torch.arange(0, batch_size + 1, mtp_step + 1, dtype=torch.int32, device=device)
            # shape 为 [att_batch_size]
            self.b_conv_buffer_idx = self.infer_state.b_req_idx.view(att_batch_size, mtp_step + 1)[:, 0].contiguous()
            self.b_ssm_buffer_idx = (self.b_conv_buffer_idx * (mtp_step + 1)).view(att_batch_size, 1) + torch.arange(
                mtp_step + 1, device=device, dtype=self.infer_state.b_req_idx.dtype
            ).view(1, mtp_step + 1)
            # shape 为 [att_batch_size]
            # 上一步接受的数量，用于linear att 的decode mtp 算子定位正确的conv 和 ssm信息的起点。
            self.b_num_accepted_tokens = self.infer_state.req_manager.req_to_mtp_state_index[self.b_conv_buffer_idx] + 1
            return

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
        backend: LinearAttBackend = self.backend

        mixed_qkv, z, b, a = backend._split_qkvzba(mixed_qkvzba)
        conv_states, ssm_states = self.infer_state.req_manager.get_mamba_cache(layer_num)

        if backend.mtp_step > 0:
            # MTP 模式下，使用线性层 MTP 状态。
            core_attn_out = self._gdn_mtp_kernel(
                mixed_qkv,
                conv_states,
                ssm_states,
                a,
                b,
                self.infer_state,
                layer_weight,
            )
        else:
            # 非 MTP 模式下，使用线性层 decode 状态。
            core_attn_out, z = self._gdn_decode_kernel(
                mixed_qkv,
                z,
                conv_states,
                ssm_states,
                a,
                b,
                self.infer_state,
                layer_weight,
            )
        return core_attn_out, z

    def _gdn_decode_kernel(
        self,
        mixed_qkv: torch.Tensor,
        z: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        infer_state: "Qwen3NextInferStateInfo",
        layer_weight: "Qwen3NextTransformerLayerWeight",
    ):
        backend: LinearAttBackend = self.backend

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
            self.b_conv_buffer_idx,
            backend.activation,
            backend.conv_kernel_dim,
            backend.tp_num_k_heads,
            backend.head_k_dim,
            backend.tp_num_v_heads,
            backend.head_v_dim,
        )
        core_attn_out, _ = fused_recurrent_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            initial_state=ssm_states,
            inplace_final_state=True,
            ssm_state_indices=self.b_ssm_buffer_idx,
            use_qk_l2norm_in_kernel=True,
            A_log=layer_weight.linear_A_log.weight,
            dt_bias=layer_weight.linear_dt_bias.weight,
            a_raw=a,
            b_raw=b,
        )
        return core_attn_out, z

    def _gdn_mtp_kernel(
        self,
        mixed_qkv: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        infer_state: "Qwen3NextInferStateInfo",
        layer_weight: "Qwen3NextTransformerLayerWeight",
    ):
        from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d_spec import (
            causal_conv1d_update as causal_conv1d_update_spec,
        )

        backend: LinearAttBackend = self.backend

        cu_seqlens_q = self.b1_mtp_cu_q_seq_len
        mixed_qkv = causal_conv1d_update_spec(
            mixed_qkv,
            conv_states,
            layer_weight.linear_conv1d.mm_param.weight,
            mtp_step=backend.mtp_step,
            bias=layer_weight.linear_conv1d.bias,
            activation=backend.activation,
            conv_state_indices=self.b_conv_buffer_idx,
            num_accepted_tokens=self.b_num_accepted_tokens,
            query_start_loc=cu_seqlens_q,
        )

        query, key, value = backend._rearrange_mixed_qkv(mixed_qkv, decode=False)
        assert self.b_ssm_buffer_idx.dim() == 2, "SSM buffer idx must be 2D [N, S+1]"
        # #8b: b_num_accepted_tokens >= 1 is guaranteed upstream: init/cache restore set 1,
        # and MTP decode only writes values in [1, mtp_step+1]. The old per-layer per-step
        # .all() D2H sync stalled the GPU on the eager decode hot path; it is redundant here.
        core_attn_out, _ = mtp_fused_recurrent_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            initial_state=ssm_states,
            cu_seqlens=cu_seqlens_q.to(torch.long),
            ssm_state_indices=self.b_ssm_buffer_idx,
            ssm_state_write_indices=self.b_ssm_buffer_idx,
            num_accepted_tokens=self.b_num_accepted_tokens,
            A_log=layer_weight.linear_A_log.weight,
            dt_bias=layer_weight.linear_dt_bias.weight,
            a_raw=a,
            b_raw=b,
        )
        return core_attn_out
