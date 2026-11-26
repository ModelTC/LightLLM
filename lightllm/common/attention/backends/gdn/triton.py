"""GDN 的 Triton prefill 和原生 SSM decode 实现。"""
from collections.abc import Callable

from lightllm.common.basemodel.infer_struct import InferStateInfo

from dataclasses import dataclass
from typing import Optional

import torch

from lightllm.common.attention.base_backend import AttRuntimeConfig, DecodeAttBackend, PrefillAttBackend
from . import GDNAttConfig, GDNAttInputs


@dataclass(frozen=True)
class GDNPrefillAttState:
    """打包prefill：请求 i 的令牌从 cu_seqlens[i] 开始；其卷积状态是 slot conv_idx[i]，其 SSM 状态槽是 ssm_idx[i]；has_initial_state[i]
    表示它是否延续了之前的令牌。
    """

    conv_idx: torch.Tensor
    ssm_idx: torch.Tensor
    cu_seqlens: torch.Tensor
    has_initial_state: torch.Tensor


@dataclass(frozen=True)
class GDNDecodeAttState:
    """按请求分组解码行。

    没有投机解码（draft_step 0）时，每行是一个请求，具有 conv 和 SSM 槽 conv_idx/ssm_idx，对于原生状态，cu_seqlens 为 None。当 draft_step > 0 时，
    cu_seqlens 给出每个请求的行数，conv_idx 给出其卷积槽，num_accepted_tokens 表示它接受的上一步草稿的数量，ssm_idx 给出其 SSM 槽（对于原生状态为
    [requests, draft_step + 1]，每个草稿保留一个；对于重放状态为 [requests]）。重放解码始终具有 cu_seqlens。
    """

    conv_idx: torch.Tensor
    ssm_idx: torch.Tensor
    cu_seqlens: Optional[torch.Tensor]
    num_accepted_tokens: Optional[torch.Tensor]


class TritonGDNPrefillBackend(PrefillAttBackend[GDNAttConfig, GDNPrefillAttState, GDNAttInputs]):
    """分块prefill，使用树内 FLA 内核；子类交换 chunk_gated_delta_rule。"""

    def __init__(
        self,
        att_config: GDNAttConfig,
        *,
        role: str = "main",
        runtime_config: AttRuntimeConfig = AttRuntimeConfig(),
    ) -> None:
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        # 现在解析块内核，因此缺少或损坏的可选库会被拒绝。
        # 在模型加载时检查，自动选择可继续尝试其余实现，避免在推理步骤中报错。
        try:
            self.chunk_gated_delta_rule()
        except Exception as e:
            raise NotImplementedError(f"{type(self).__name__} cannot load its chunk kernel: {e!r}") from e

    @staticmethod
    def chunk_gated_delta_rule():
        from lightllm.common.attention.kernels.gdn.third_party_fla.ops.chunk import chunk_gated_delta_rule

        return chunk_gated_delta_rule

    def prefill(
        self,
        att_state: GDNPrefillAttState,
        layer_inputs: GDNAttInputs,
        *,
        out: Optional[torch.Tensor],
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """返回块输出，如果提供了 out 则复制到 out 中，并就地更新卷积和 SSM 状态。"""
        from lightllm.common.attention.kernels.gdn.causal_conv1d import causal_conv1d_fn
        from lightllm.common.attention.kernels.gdn.fused_gdn_gating import fused_gdn_gating

        cfg, inputs = self.att_config, layer_inputs
        conv_states = inputs.conv_states
        if cfg.state_mtp_step > 0:
            # Prefill 只读取卷积窗口。
            conv_states = conv_states[..., : -cfg.state_mtp_step]
        g, beta = fused_gdn_gating(inputs.A_log, inputs.a, inputs.b, inputs.dt_bias)
        mixed_qkv = causal_conv1d_fn(
            x=inputs.mixed_qkv.transpose(0, 1),
            weight=inputs.conv_weight,
            bias=inputs.conv_bias,
            query_start_loc=att_state.cu_seqlens,
            cache_indices=att_state.conv_idx,
            has_initial_state=att_state.has_initial_state,
            conv_states=conv_states,
            activation=cfg.activation,
        ).transpose(0, 1)
        query, key, value = TritonGDNPrefillBackend._split_qkv(att_config=cfg, mixed_qkv=mixed_qkv)
        tokens = query.shape[0]
        ssm_states = inputs.ssm_states
        core, last_state = self.chunk_gated_delta_rule()(
            q=query.view(1, tokens, cfg.num_k_heads, cfg.head_k_dim),
            k=key.view(1, tokens, cfg.num_k_heads, cfg.head_k_dim),
            v=value.view(1, tokens, cfg.num_v_heads, cfg.head_v_dim),
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=ssm_states[att_state.ssm_idx],
            output_final_state=True,
            cu_seqlens=att_state.cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )
        # 块内核在 float32 中累积；index_put 不会隐式转换数据类型。
        ssm_states[att_state.ssm_idx] = last_state.to(ssm_states.dtype, copy=False)
        output = core.view(tokens, cfg.num_v_heads, cfg.head_v_dim)
        if out is None:
            return output
        out.copy_(output)
        return out

    @staticmethod
    def get_name() -> str:
        return "triton_gdn_prefill"

    def create_att_state(self, infer_state: InferStateInfo) -> GDNPrefillAttState:
        req = infer_state.b_req_idx
        slots = 1 if self.att_config.replay else self.att_config.state_mtp_step + 1
        return GDNPrefillAttState(
            conv_idx=req,
            ssm_idx=req * slots if slots != 1 else req,
            cu_seqlens=infer_state.b1_cu_q_seq_len,
            has_initial_state=infer_state.b_ready_cache_len > 0,
        )

    @staticmethod
    def _split_qkv(att_config: GDNAttConfig, mixed_qkv: torch.Tensor):
        key_dim = att_config.num_k_heads * att_config.head_k_dim
        value_dim = att_config.num_v_heads * att_config.head_v_dim
        return torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)


class TritonGDNDecodeBackend(DecodeAttBackend[GDNAttConfig, GDNDecodeAttState, GDNAttInputs]):
    def __init__(
        self,
        att_config: GDNAttConfig,
        *,
        role: str = "main",
        runtime_config: AttRuntimeConfig = AttRuntimeConfig(),
    ) -> None:
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        if att_config.replay:
            raise NotImplementedError("Triton GDN decode updates native SSM states; replay state needs replay decode")

    def decode(
        self,
        att_state: GDNDecodeAttState,
        layer_inputs: GDNAttInputs,
        *,
        out: torch.Tensor,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """填充并返回 out [rows, num_v_heads, head_v_dim]；就地更新卷积和 SSM 状态。"""
        cfg, inputs = self.att_config, layer_inputs
        if self.runtime_config.decode_rows_per_request == 1:
            from lightllm.common.attention.kernels.gdn.third_party_fla.ops.fused_recurrent import (
                fused_recurrent_gated_delta_rule,
            )

            query, key, value, a, b = TritonGDNDecodeBackend._pack_decode_inputs(
                cfg=cfg, inputs=inputs, conv_idx=att_state.conv_idx
            )
            fused_recurrent_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                initial_state=inputs.ssm_states,
                inplace_final_state=True,
                ssm_state_indices=att_state.ssm_idx,
                use_qk_l2norm_in_kernel=True,
                A_log=inputs.A_log,
                dt_bias=inputs.dt_bias,
                a_raw=a,
                b_raw=b,
                out=out.view(value.shape),
            )
        else:
            from lightllm.common.attention.kernels.gdn.mtp_fused_recurrent import mtp_fused_recurrent_gated_delta_rule

            mixed_qkv = TritonGDNDecodeBackend._mtp_conv(
                cfg=cfg, inputs=inputs, att_state=att_state, draft_step=self.runtime_config.decode_rows_per_request - 1
            )
            query, key, value = TritonGDNPrefillBackend._split_qkv(att_config=cfg, mixed_qkv=mixed_qkv)
            rows = query.shape[0]
            # num_accepted_tokens >= 1 上游保证：缓存恢复设置为 1，MTP 解码
            # 仅在 [1, draft_step + 1] 范围内写入值。
            values = value.view(1, rows, cfg.num_v_heads, cfg.head_v_dim)
            mtp_fused_recurrent_gated_delta_rule(
                q=query.view(1, rows, cfg.num_k_heads, cfg.head_k_dim),
                k=key.view(1, rows, cfg.num_k_heads, cfg.head_k_dim),
                v=values,
                initial_state=inputs.ssm_states,
                cu_seqlens=att_state.cu_seqlens.to(torch.long),
                ssm_state_indices=att_state.ssm_idx,
                ssm_state_write_indices=att_state.ssm_idx,
                num_accepted_tokens=att_state.num_accepted_tokens,
                A_log=inputs.A_log,
                dt_bias=inputs.dt_bias,
                a_raw=inputs.a,
                b_raw=inputs.b,
                out=out.view(values.shape),
            )
        return out

    @staticmethod
    def get_name() -> str:
        return "triton_gdn_decode"

    def create_att_state(self, infer_state: InferStateInfo) -> GDNDecodeAttState:
        mtp_index = None
        if self.runtime_config.decode_rows_per_request > 1:
            state = infer_state.att_state_manager.get_state(
                model_role=self.role, state_type_filter="gdn", logical_layer_id=self.logical_layer_ids[0]
            )
            mtp_index = state.mtp_state_index
        return TritonGDNDecodeBackend.prepare_native_decode_state(
            infer_state=infer_state, runtime_config=self.runtime_config, req_to_mtp_state_index=mtp_index
        )

    @staticmethod
    def prepare_mtp_decode_rows(
        infer_state: InferStateInfo,
        runtime_config: AttRuntimeConfig,
        req_to_mtp_state_index: torch.Tensor,
    ):
        """构造投机解码的请求槽、行偏移和上一步接受的 token 数。"""
        from lightllm.common.attention.kernels.gdn.mtp_state_params import build_dynamic_mtp_linear_att_state_params

        req = infer_state.b_req_idx
        width = runtime_config.decode_rows_per_request
        assert req_to_mtp_state_index is not None, "speculative GDN decode needs req_to_mtp_state_index"
        if runtime_config.dynamic_decode_rows:
            cu, conv_idx, accepted = build_dynamic_mtp_linear_att_state_params(
                b_req_idx=req,
                b_mtp_index=infer_state.b_mtp_index,
                req_to_mtp_state_index=req_to_mtp_state_index,
                hold_req_id=infer_state.pad_req_idx,
            )
        else:
            assert infer_state.batch_size % width == 0, (
                "GDN fixed-layout decode requires batch_size to be divisible by draft_step + 1, "
                f"got batch_size={infer_state.batch_size}, draft_step={width - 1}."
            )
            offsets = TritonGDNDecodeBackend.prepare_mtp_decode_rows.__dict__.setdefault("_query_offsets", {})
            key = (req.device, infer_state.batch_size, width)
            cu = offsets.get(key)
            if cu is None:
                cu = offsets.setdefault(
                    key,
                    torch.arange(
                        start=0, end=infer_state.batch_size + 1, step=width, dtype=torch.int32, device=req.device
                    ),
                )
            conv_idx = req.view(-1, width)[:, 0].contiguous()
            accepted = req_to_mtp_state_index[conv_idx] + 1
        return cu, conv_idx, accepted

    @staticmethod
    def prepare_native_decode_state(
        infer_state: InferStateInfo,
        runtime_config: AttRuntimeConfig,
        req_to_mtp_state_index: Optional[torch.Tensor],
    ) -> GDNDecodeAttState:
        req = infer_state.b_req_idx
        width = runtime_config.decode_rows_per_request
        if width == 1:
            return GDNDecodeAttState(conv_idx=req, ssm_idx=req, cu_seqlens=None, num_accepted_tokens=None)

        cu, conv_idx, accepted = TritonGDNDecodeBackend.prepare_mtp_decode_rows(
            infer_state=infer_state, runtime_config=runtime_config, req_to_mtp_state_index=req_to_mtp_state_index
        )
        # 原生状态为每个请求保留 width 个连续的 SSM 槽。
        ssm_offsets = TritonGDNDecodeBackend.prepare_native_decode_state.__dict__.setdefault("_ssm_offsets", {})
        key = (req.device, req.dtype, width)
        offsets = ssm_offsets.get(key)
        if offsets is None:
            offsets = ssm_offsets.setdefault(key, torch.arange(end=width, device=req.device, dtype=req.dtype))
        ssm_idx = (conv_idx * width).view(-1, 1) + offsets
        return GDNDecodeAttState(conv_idx=conv_idx, ssm_idx=ssm_idx, cu_seqlens=cu, num_accepted_tokens=accepted)

    @staticmethod
    def _pack_decode_inputs(cfg: GDNAttConfig, inputs: GDNAttInputs, conv_idx: torch.Tensor):
        from lightllm.common.attention.kernels.gdn.gdn_decode_pack import conv_pack_gdn_decode_inputs

        # 融合卷积更新和 q/k/v/gate 打包，不生成卷积后的 qkv。
        return conv_pack_gdn_decode_inputs(
            inputs.mixed_qkv,
            inputs.a,
            inputs.b,
            inputs.conv_states,
            inputs.conv_weight,
            inputs.conv_bias,
            conv_idx,
            cfg.activation,
            cfg.conv_kernel_dim,
            cfg.num_k_heads,
            cfg.head_k_dim,
            cfg.num_v_heads,
            cfg.head_v_dim,
        )

    @staticmethod
    def _mtp_conv(cfg: GDNAttConfig, inputs: GDNAttInputs, att_state: GDNDecodeAttState, draft_step: int):
        from lightllm.common.attention.kernels.gdn.causal_conv1d_mtp import causal_conv1d_update

        return causal_conv1d_update(
            x=inputs.mixed_qkv,
            conv_state=inputs.conv_states,
            weight=inputs.conv_weight,
            mtp_step=draft_step,
            bias=inputs.conv_bias,
            activation=cfg.activation,
            conv_state_indices=att_state.conv_idx,
            num_accepted_tokens=att_state.num_accepted_tokens,
            query_start_loc=att_state.cu_seqlens,
        )
