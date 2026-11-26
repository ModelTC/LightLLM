# SPDX-License-Identifier: Apache-2.0

"""KDA 实现：分块prefill（Triton，或短步骤的 FlashKDA）和在原生或重放递归状态上的解码。解码行按请求分组。"""

from lightllm.common.basemodel.infer_struct import InferStateInfo

from functools import cached_property
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from lightllm.common.attention.base_backend import AttRuntimeConfig, DecodeAttBackend, PrefillAttBackend
from ..gdn.replay import ReplayGDNDecodeBackend
from ..gdn.triton import GDNDecodeAttState, TritonGDNDecodeBackend
from lightllm.utils.log_utils import init_logger
from . import KDAAttConfig, KDAAttInputs

logger = init_logger(name=__name__)


@dataclass(frozen=True)
class KDAPrefillAttState:
    """打包后的 prefill 状态。

    请求 i 的 token 从 cu_seqlens[i] 开始。FlashKDA 路径的 cu_seqlens 使用 int64。
    卷积和 SSM 状态槽分别为 conv_idx[i] 和 ssm_idx[i]，本步前已缓存
    b_ready_cache_len[i] 个 token。flash_kda_fwd 是本步选中的 FlashKDA kernel，
    使用 Triton 时为 None。
    """

    conv_idx: torch.Tensor
    ssm_idx: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens: torch.Tensor
    b_ready_cache_len: torch.Tensor
    max_q_seq_len: int
    flash_kda_fwd: Optional[Callable]


class TritonKDAPrefillBackend(PrefillAttBackend[KDAAttConfig, KDAPrefillAttState, KDAAttInputs]):
    """在原生状态上的prefill：融合 conv 发出 q/k/v（除非 FlashKDA 运行，否则为 L2 归一化）。"""

    def __init__(
        self,
        att_config: KDAAttConfig,
        *,
        role: str = "main",
        runtime_config: AttRuntimeConfig = AttRuntimeConfig(),
    ) -> None:
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        TritonKDAPrefillBackend._check_state(att_config=att_config, replay=False)
        TritonKDAPrefillBackend._check_prefill_kernel(att_config=att_config)

    def prefill(
        self,
        att_state: KDAPrefillAttState,
        layer_inputs: KDAAttInputs,
        *,
        out: torch.Tensor | None,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ):
        from lightllm.common.attention.kernels.kda.causal_conv1d_qkv import causal_conv1d_qkv

        cfg, inputs = self.att_config, layer_inputs
        flash = att_state.flash_kda_fwd is not None
        q, k, v = causal_conv1d_qkv(
            x=inputs.mixed_qkv,
            weight=inputs.conv_weight,
            conv_states=inputs.conv_states[..., : cfg.conv_kernel_size - 1],
            query_start_loc=att_state.cu_seqlens_q,
            cache_indices=att_state.conv_idx,
            cache_seqlens=att_state.b_ready_cache_len,
            max_q_seq_len=att_state.max_q_seq_len,
            use_qk_l2norm=not flash,
        )
        return TritonKDAPrefillBackend._prefill(
            cfg=cfg, att_state=att_state, inputs=inputs, q=q, k=k, v=v, qk_normalized=not flash, out=out
        )

    @staticmethod
    def get_name() -> str:
        return "triton_kda_prefill"

    def create_att_state(self, infer_state: InferStateInfo) -> KDAPrefillAttState:
        ssm_idx = infer_state.b_req_idx * (self.att_config.state_mtp_step + 1)
        return TritonKDAPrefillBackend._prepare_prefill_state(
            infer_state=infer_state, att_config=self.att_config, ssm_idx=ssm_idx, flash_kda_fwd=self._flash_kda_fwd
        )

    @staticmethod
    def import_flash_kda_fwd() -> Callable:
        """Import FlashKDA's fwd; the ImportError says how to install it."""
        try:
            from flash_kda import fwd
        except (ImportError, OSError) as exc:
            raise ImportError(
                "The flashkda prefill backend requires FlashKDA. Install it with "
                "`pip install --no-build-isolation git+https://github.com/MoonshotAI/FlashKDA.git` "
                "and build it for the current GPU and PyTorch/CUDA versions."
            ) from exc
        return fwd

    @staticmethod
    def load_flash_kda_fwd(requested: str, *, dtype: torch.dtype, ssm_dtype: torch.dtype) -> Optional[Callable]:
        """Return FlashKDA's fwd for the requested prefill provider, or None to use Triton.

        requested is "auto", "flashkda" or "triton". "auto" uses FlashKDA only for BF16 models with a
        BF16 SSM state on SM90 when it loads; "flashkda" raises if it cannot be loaded.
        """
        if requested == "triton":
            return None
        if requested not in ("auto", "flashkda"):
            raise ValueError(f"Unsupported KDA prefill backend: {requested}; use auto, triton, or flashkda")
        if requested == "auto" and (
            dtype != torch.bfloat16
            or ssm_dtype != torch.bfloat16
            or torch.version.hip is not None
            or not torch.cuda.is_available()
            or torch.cuda.get_device_capability() != (9, 0)
        ):
            return None
        try:
            fwd = TritonKDAPrefillBackend.import_flash_kda_fwd()
        except ImportError as exc:
            if requested == "auto":
                logger.info("KDA prefill uses Triton because FlashKDA could not be loaded: %s", exc.__cause__)
                return None
            raise
        logger.info(
            "KDA prefill attention backend: %s.", "automatic FlashKDA/Triton" if requested == "auto" else "flashkda"
        )
        return fwd

    @cached_property
    def _flash_kda_fwd(self):
        config = self.att_config
        return TritonKDAPrefillBackend.load_flash_kda_fwd(
            requested=config.prefill_kernel, dtype=config.dtype, ssm_dtype=config.ssm_dtype
        )

    @staticmethod
    def _prepare_prefill_state(
        infer_state: InferStateInfo, att_config: KDAAttConfig, ssm_idx: torch.Tensor, flash_kda_fwd: Optional[Callable]
    ) -> KDAPrefillAttState:
        # 自动选择将 FlashKDA 限制在已验证的请求长度上。
        max_flash_len = 2048 if att_config.num_heads < 16 else 8192
        if (
            flash_kda_fwd is not None
            and att_config.prefill_kernel == "auto"
            and infer_state.max_q_seq_len > max_flash_len
        ):
            flash_kda_fwd = None
        cu = infer_state.b1_cu_q_seq_len
        return KDAPrefillAttState(
            conv_idx=infer_state.b_req_idx,
            ssm_idx=ssm_idx,
            cu_seqlens_q=cu,
            cu_seqlens=cu.to(torch.int64).contiguous() if flash_kda_fwd is not None else cu,
            b_ready_cache_len=infer_state.b_ready_cache_len,
            max_q_seq_len=infer_state.max_q_seq_len,
            flash_kda_fwd=flash_kda_fwd,
        )

    @staticmethod
    def _prefill(
        cfg: KDAAttConfig,
        att_state: KDAPrefillAttState,
        inputs: KDAAttInputs,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qk_normalized: bool,
        out: torch.Tensor | None,
    ):
        from lightllm.common.attention.kernels.kda.kda_prefill import kda_prefill

        result = kda_prefill(
            q=q,
            k=k,
            v=v,
            raw_g=inputs.raw_gate.view(1, -1, cfg.num_heads, cfg.head_dim),
            raw_beta=inputs.raw_beta.view(1, -1, cfg.num_heads),
            A_log=inputs.A_log,
            g_bias=inputs.dt_bias,
            ssm_states=inputs.ssm_states,
            state_indices=att_state.ssm_idx,
            qk_normalized=qk_normalized,
            cu_seqlens=att_state.cu_seqlens,
            lower_bound=cfg.lower_bound,
            flash_kda_fwd=att_state.flash_kda_fwd,
            out=None if out is None else out.view(v.shape),
        )
        return result.squeeze(0) if out is None else out

    @staticmethod
    def _check_state(att_config: KDAAttConfig, *, replay: bool) -> None:
        if att_config.replay != replay:
            state = "replay" if att_config.replay else "native"
            raise NotImplementedError(f"this KDA implementation does not run over {state} state")

    @staticmethod
    def _check_prefill_kernel(att_config: KDAAttConfig) -> None:
        # 请求的 FlashKDA 必须在模型加载时加载，而不是在第一次prefill步骤时加载。
        if att_config.prefill_kernel == "flashkda":
            try:
                TritonKDAPrefillBackend.import_flash_kda_fwd()
            except ImportError as e:
                raise NotImplementedError(str(e)) from e


class ReplayKDAPrefillBackend(PrefillAttBackend[KDAAttConfig, KDAPrefillAttState, KDAAttInputs]):
    """在重放状态上的prefill，该状态保留原始 q/k：融合 conv 仅在 FlashKDA 宽度为 4 时运行。"""

    def __init__(
        self,
        att_config: KDAAttConfig,
        *,
        role: str = "main",
        runtime_config: AttRuntimeConfig = AttRuntimeConfig(),
    ) -> None:
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        TritonKDAPrefillBackend._check_state(att_config=att_config, replay=True)
        TritonKDAPrefillBackend._check_prefill_kernel(att_config=att_config)

    def prefill(
        self,
        att_state: KDAPrefillAttState,
        layer_inputs: KDAAttInputs,
        *,
        out: torch.Tensor | None,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ):
        cfg, inputs = self.att_config, layer_inputs
        conv_states = inputs.conv_states[..., : cfg.conv_kernel_size - 1]
        if att_state.flash_kda_fwd is not None and cfg.conv_kernel_size == 4:
            from lightllm.common.attention.kernels.kda.causal_conv1d_qkv import causal_conv1d_qkv

            q, k, v = causal_conv1d_qkv(
                inputs.mixed_qkv,
                inputs.conv_weight,
                conv_states,
                att_state.cu_seqlens_q,
                att_state.conv_idx,
                att_state.b_ready_cache_len,
                att_state.max_q_seq_len,
                use_qk_l2norm=False,
            )
        else:
            from lightllm.common.attention.kernels.gdn.causal_conv1d import causal_conv1d_fn

            mixed_qkv = causal_conv1d_fn(
                inputs.mixed_qkv.transpose(0, 1),
                inputs.conv_weight,
                bias=None,
                query_start_loc=att_state.cu_seqlens_q,
                cache_indices=att_state.conv_idx,
                has_initial_state=att_state.b_ready_cache_len > 0,
                conv_states=conv_states,
                activation="silu",
            ).transpose(0, 1)
            q, k, v = (
                x.view(1, -1, cfg.num_heads, cfg.head_dim) for x in mixed_qkv.split(cfg.num_heads * cfg.head_dim, -1)
            )
        return TritonKDAPrefillBackend._prefill(
            cfg=cfg, att_state=att_state, inputs=inputs, q=q, k=k, v=v, qk_normalized=False, out=out
        )

    @staticmethod
    def get_name() -> str:
        return "replay_kda_prefill"

    def create_att_state(self, infer_state: InferStateInfo) -> KDAPrefillAttState:
        ssm_idx = infer_state.b_req_idx
        return TritonKDAPrefillBackend._prepare_prefill_state(
            infer_state=infer_state, att_config=self.att_config, ssm_idx=ssm_idx, flash_kda_fwd=self._flash_kda_fwd
        )

    @cached_property
    def _flash_kda_fwd(self):
        config = self.att_config
        return TritonKDAPrefillBackend.load_flash_kda_fwd(
            requested=config.prefill_kernel, dtype=config.dtype, ssm_dtype=config.ssm_dtype
        )


class TritonKDADecodeBackend(DecodeAttBackend[KDAAttConfig, GDNDecodeAttState, KDAAttInputs]):
    def __init__(
        self,
        att_config: KDAAttConfig,
        *,
        role: str = "main",
        runtime_config: AttRuntimeConfig = AttRuntimeConfig(),
    ) -> None:
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        TritonKDAPrefillBackend._check_state(att_config=att_config, replay=False)

    def decode(
        self,
        att_state: GDNDecodeAttState,
        layer_inputs: KDAAttInputs,
        *,
        out: torch.Tensor | None,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ):
        """填充并返回 out [rows, num_heads, head_dim]；就地更新 conv 和 SSM 状态。"""
        from lightllm.common.attention.kernels.gdn.third_party_fla.ops.kda_decode import fused_recurrent_kda

        cfg, inputs = self.att_config, layer_inputs
        hidden = cfg.num_heads * cfg.head_dim
        q, k, v = TritonKDADecodeBackend._conv_update(
            inputs=inputs, att_state=att_state, draft_step=self.runtime_config.decode_rows_per_request - 1
        ).split(hidden, dim=-1)
        shape = (1, -1) if self.runtime_config.decode_rows_per_request > 1 else (-1, 1)
        fused_recurrent_kda(
            q=q.view(*shape, cfg.num_heads, cfg.head_dim),
            k=k.view(*shape, cfg.num_heads, cfg.head_dim),
            v=v.view(*shape, cfg.num_heads, cfg.head_dim),
            raw_gate=inputs.raw_gate.view(*shape, hidden),
            raw_beta=inputs.raw_beta.view(*shape, cfg.num_heads),
            a_log=inputs.A_log,
            gate_bias=inputs.dt_bias,
            initial_state=inputs.ssm_states,
            lower_bound=cfg.lower_bound,
            inplace_final_state=True,
            ssm_state_indices=att_state.ssm_idx,
            cu_seqlens=att_state.cu_seqlens,
            num_accepted_tokens=att_state.num_accepted_tokens,
            out=out.view(*shape, cfg.num_heads, cfg.head_dim),
        )
        return out

    @staticmethod
    def get_name() -> str:
        return "triton_kda_decode"

    def create_att_state(self, infer_state: InferStateInfo) -> GDNDecodeAttState:
        mtp_index = None
        if self.runtime_config.decode_rows_per_request > 1:
            state = infer_state.att_state_manager.get_state(
                model_role=self.role, state_type_filter="kda", logical_layer_id=self.logical_layer_ids[0]
            )
            mtp_index = state.mtp_state_index
        return TritonGDNDecodeBackend.prepare_native_decode_state(
            infer_state=infer_state, runtime_config=self.runtime_config, req_to_mtp_state_index=mtp_index
        )

    @staticmethod
    def _conv_update(inputs: KDAAttInputs, att_state: GDNDecodeAttState, draft_step: int):
        from lightllm.common.attention.kernels.gdn.causal_conv1d import causal_conv1d_update
        from lightllm.common.attention.kernels.gdn.causal_conv1d_mtp import (
            causal_conv1d_update as causal_conv1d_update_mtp,
        )

        if draft_step > 0:
            return causal_conv1d_update_mtp(
                x=inputs.mixed_qkv,
                conv_state=inputs.conv_states,
                weight=inputs.conv_weight,
                mtp_step=draft_step,
                activation="silu",
                conv_state_indices=att_state.conv_idx,
                num_accepted_tokens=att_state.num_accepted_tokens,
                query_start_loc=att_state.cu_seqlens,
            )
        return causal_conv1d_update(
            x=inputs.mixed_qkv,
            conv_state=inputs.conv_states,
            weight=inputs.conv_weight,
            activation="silu",
            conv_state_indices=att_state.conv_idx,
        )


class ReplayKDADecodeBackend(DecodeAttBackend[KDAAttConfig, GDNDecodeAttState, KDAAttInputs]):
    def __init__(
        self,
        att_config: KDAAttConfig,
        *,
        role: str = "main",
        runtime_config: AttRuntimeConfig = AttRuntimeConfig(),
    ) -> None:
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        TritonKDAPrefillBackend._check_state(att_config=att_config, replay=True)

    def decode(
        self,
        att_state: GDNDecodeAttState,
        layer_inputs: KDAAttInputs,
        *,
        out: torch.Tensor | None,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ):
        """填充并返回 out [rows, num_heads, head_dim]；更新 conv 状态并追加历史记录。

        每一层读取步骤开始时的游标；最后一层的调用会推进它们。
        """
        from lightllm.common.attention.kernels.kda.replay_kda import replay

        cfg, inputs = self.att_config, layer_inputs
        hidden = cfg.num_heads * cfg.head_dim
        q, k, v = (
            x.view(-1, cfg.num_heads, cfg.head_dim)
            for x in TritonKDADecodeBackend._conv_update(
                inputs=inputs, att_state=att_state, draft_step=self.runtime_config.decode_rows_per_request - 1
            ).split(hidden, -1)
        )
        state = inputs.replay_state
        layer = state.storage_layer(logical_layer_id=inputs.logic_layer_index)
        replay(
            state_ssm=state.ssm_state[layer],
            state_keys=state.keys[layer],
            state_deltas=state.deltas[layer],
            state_gates=state.gates[layer],
            state_raw_keys=state.raw_keys[layer],
            state_raw_values=state.raw_values[layer],
            state_betas=state.betas[layer],
            state_cursors=state.cursors,
            state_tile_sync_mark=state.tile_sync_mark,
            q=q,
            k=k,
            v=v,
            a=inputs.raw_gate.reshape(-1, hidden),
            b=inputs.raw_beta.reshape(-1, cfg.num_heads),
            a_log=inputs.A_log,
            bias=inputs.dt_bias,
            reqs=att_state.conv_idx,
            cu_seqlens=att_state.cu_seqlens,
            state_verify_width=state.verify_width,
            state_compute_mode=state.compute_mode,
            state_gate_lower_bound=state.gate_lower_bound,
            out=out.view(v.shape),
        )
        state.finish_decode_token(
            logical_layer_id=inputs.logic_layer_index, reqs=att_state.conv_idx, cu_seqlens=att_state.cu_seqlens
        )
        return out

    @staticmethod
    def get_name() -> str:
        return "replay_kda_decode"

    def create_att_state(self, infer_state: InferStateInfo) -> GDNDecodeAttState:
        mtp_index = None
        if self.runtime_config.decode_rows_per_request > 1:
            state = infer_state.att_state_manager.get_state(
                model_role=self.role, state_type_filter="kda", logical_layer_id=self.logical_layer_ids[0]
            )
            mtp_index = state.mtp_state_index
        return ReplayGDNDecodeBackend.prepare_replay_decode_state(
            infer_state=infer_state, runtime_config=self.runtime_config, req_to_mtp_state_index=mtp_index
        )
