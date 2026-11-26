"""GDN 解码重放状态：不再每令牌更新 SSM 状态，而是将行追加到每层的历史记录中，重放内核从每个请求的游标读取这些历史记录。prefill是常规分块prefill，每个请求有一个 SSM 位置。"""
from collections.abc import Callable

from lightllm.common.basemodel.infer_struct import InferStateInfo

from lightllm.common.attention.base_backend import AttRuntimeConfig, DecodeAttBackend

import torch

from . import GDNAttConfig, GDNAttInputs
from .triton import GDNDecodeAttState, TritonGDNDecodeBackend, TritonGDNPrefillBackend


class ReplayGDNDecodeBackend(DecodeAttBackend[GDNAttConfig, GDNDecodeAttState, GDNAttInputs]):
    def __init__(
        self,
        att_config: GDNAttConfig,
        *,
        role: str = "main",
        runtime_config: AttRuntimeConfig = AttRuntimeConfig(),
    ) -> None:
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        if not att_config.replay:
            raise NotImplementedError("replay GDN decode needs replay state")

    def decode(
        self,
        att_state: GDNDecodeAttState,
        layer_inputs: GDNAttInputs,
        *,
        out: torch.Tensor,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """填充并返回 out [行数, num_v_heads, head_v_dim]；更新卷积状态并追加历史记录。

        每层读取该步骤开始时的游标；最后一层的调用会推进它们。
        """
        from lightllm.common.attention.kernels.gdn.replayssm.replay_gdn import replay

        cfg, inputs = self.att_config, layer_inputs
        state = inputs.replay_state
        logic_layer_index = inputs.logic_layer_index
        assert (
            state is not None and logic_layer_index is not None
        ), "replay decode needs replay_state and logic_layer_index"
        if self.runtime_config.decode_rows_per_request == 1:
            query, key, value, a, b = TritonGDNDecodeBackend._pack_decode_inputs(
                cfg=cfg, inputs=inputs, conv_idx=att_state.conv_idx
            )
            query = query.view(query.shape[0], cfg.num_k_heads, cfg.head_k_dim)
            key = key.view(key.shape[0], cfg.num_k_heads, cfg.head_k_dim)
            value = value.view(value.shape[0], cfg.num_v_heads, cfg.head_v_dim)
        else:
            query, key, value = TritonGDNPrefillBackend._split_qkv(
                att_config=cfg,
                mixed_qkv=TritonGDNDecodeBackend._mtp_conv(
                    cfg=cfg,
                    inputs=inputs,
                    att_state=att_state,
                    draft_step=self.runtime_config.decode_rows_per_request - 1,
                ),
            )
            query = query.view(-1, cfg.num_k_heads, cfg.head_k_dim)
            key = key.view(-1, cfg.num_k_heads, cfg.head_k_dim)
            value = value.view(-1, cfg.num_v_heads, cfg.head_v_dim)
            a, b = inputs.a, inputs.b
        replay(
            state=state,
            linear_layer_idx=state.storage_layer(logical_layer_id=logic_layer_index),
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            a_log=inputs.A_log,
            bias=inputs.dt_bias,
            reqs=att_state.conv_idx,
            cu_seqlens=att_state.cu_seqlens,
            out=out.view(value.shape),
        )
        state.finish_decode_token(
            logical_layer_id=logic_layer_index, reqs=att_state.conv_idx, cu_seqlens=att_state.cu_seqlens
        )
        return out

    @staticmethod
    def get_name() -> str:
        return "replay_gdn_decode"

    def create_att_state(self, infer_state: InferStateInfo) -> GDNDecodeAttState:
        mtp_index = None
        if self.runtime_config.decode_rows_per_request > 1:
            state = infer_state.att_state_manager.get_state(
                model_role=self.role, state_type_filter="gdn", logical_layer_id=self.logical_layer_ids[0]
            )
            mtp_index = state.mtp_state_index
        return ReplayGDNDecodeBackend.prepare_replay_decode_state(
            infer_state=infer_state, runtime_config=self.runtime_config, req_to_mtp_state_index=mtp_index
        )

    @staticmethod
    def prepare_replay_decode_state(
        infer_state: InferStateInfo,
        runtime_config: AttRuntimeConfig,
        req_to_mtp_state_index: torch.Tensor | None,
    ) -> GDNDecodeAttState:
        width = runtime_config.decode_rows_per_request
        if width > 1:
            cu, conv_idx, accepted = TritonGDNDecodeBackend.prepare_mtp_decode_rows(
                infer_state=infer_state, runtime_config=runtime_config, req_to_mtp_state_index=req_to_mtp_state_index
            )
            return GDNDecodeAttState(conv_idx=conv_idx, ssm_idx=conv_idx, cu_seqlens=cu, num_accepted_tokens=accepted)

        req = infer_state.b_req_idx
        # 重放内核始终读取行偏移，即使每个请求只有一行。
        offsets = ReplayGDNDecodeBackend.prepare_replay_decode_state.__dict__.setdefault("_query_offsets", {})
        key = (req.device, req.numel())
        cu = offsets.get(key)
        if cu is None:
            cu = offsets.setdefault(key, torch.arange(end=req.numel() + 1, dtype=torch.int32, device=req.device))
        return GDNDecodeAttState(conv_idx=req, ssm_idx=req, cu_seqlens=cu, num_accepted_tokens=None)
