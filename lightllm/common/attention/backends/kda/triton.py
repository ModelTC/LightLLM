"""KDA implementations: chunked prefill (Triton, or FlashKDA for short steps) and decode over native or
replay recurrent state. Decode rows group into requests like GDN decode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from lightllm.common.attention.backend import AttentionBackend, TensorAllocator
from ..gdn.triton import GDNDecodeAttState, build_recurrent_decode_state
from . import KDALayerConfig, KDALayerInputs
from .flashkda import import_flash_kda_fwd, load_flash_kda_fwd

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo


@dataclass(frozen=True)
class KDAPrefillAttState:
    """Packed prefill: request i's tokens start at cu_seqlens[i] (int64 when FlashKDA runs), its conv
    and SSM states are slots conv_idx[i] and ssm_idx[i], and b_ready_cache_len[i] tokens precede the
    step. flash_kda_fwd is the FlashKDA kernel chosen for this step, or None for Triton."""

    conv_idx: torch.Tensor
    ssm_idx: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens: torch.Tensor
    b_ready_cache_len: torch.Tensor
    max_q_seq_len: int
    flash_kda_fwd: Optional[Callable]


def _prefill(
    config: KDALayerConfig, att_state: KDAPrefillAttState, inputs: KDALayerInputs, q, k, v, qk_normalized, out
):
    from lightllm.common.attention.kernels.kda.kda_prefill import kda_prefill

    result = kda_prefill(
        q=q,
        k=k,
        v=v,
        raw_g=inputs.raw_gate.view(1, -1, config.num_heads, config.head_dim),
        raw_beta=inputs.raw_beta.view(1, -1, config.num_heads),
        A_log=inputs.A_log,
        g_bias=inputs.dt_bias,
        ssm_states=inputs.ssm_states,
        state_indices=att_state.ssm_idx,
        qk_normalized=qk_normalized,
        cu_seqlens=att_state.cu_seqlens,
        lower_bound=config.lower_bound,
        flash_kda_fwd=att_state.flash_kda_fwd,
        out=None if out is None else out.view(v.shape),
    )
    return result.squeeze(0) if out is None else out


def _check_state(cls, layer_config: KDALayerConfig) -> None:
    """Reject a layer whose state layout (native or replay) differs from the backend's."""
    if layer_config.replay != cls.replay:
        state = "replay" if layer_config.replay else "native"
        raise NotImplementedError(f"{cls.name} does not run over {state} state")


def _conv_update(inputs: KDALayerInputs, att_state: GDNDecodeAttState):
    from lightllm.common.attention.kernels.gdn.causal_conv1d import causal_conv1d_update
    from lightllm.common.attention.kernels.gdn.causal_conv1d_mtp import causal_conv1d_update as causal_conv1d_update_mtp

    kwargs = dict(bias=None, activation="silu", conv_state_indices=att_state.conv_idx)
    update = causal_conv1d_update
    if att_state.draft_step > 0:
        update = causal_conv1d_update_mtp
        kwargs.update(
            mtp_step=att_state.draft_step,
            num_accepted_tokens=att_state.num_accepted_tokens,
            query_start_loc=att_state.cu_seqlens,
        )
    return update(inputs.mixed_qkv, inputs.conv_states, inputs.conv_weight, **kwargs)


class TritonKDAPrefillBackend(AttentionBackend[KDALayerConfig, KDAPrefillAttState, KDALayerInputs]):
    """Prefill over native state: the fused conv emits q/k/v (L2-normalized unless FlashKDA runs)."""

    name = "triton_kda_prefill"
    phase = "prefill"
    # Which state layout the backend runs over; see KDALayerConfig.replay.
    replay = False

    def __init__(self, layer_config: KDALayerConfig) -> None:
        super().__init__(layer_config)
        # A requested FlashKDA must load when the model loads, not on the first prefill step.
        if layer_config.prefill_kernel == "flashkda":
            try:
                import_flash_kda_fwd()
            except ImportError as e:
                raise NotImplementedError(str(e)) from e

    @classmethod
    def _check_supported(cls, layer_config: KDALayerConfig) -> None:
        _check_state(cls, layer_config)

    @classmethod
    def group_key(cls, layer_config: KDALayerConfig):
        c = layer_config
        return c.replay, c.state_mtp_step, c.prefill_kernel, c.num_heads, c.dtype, c.ssm_dtype

    def init_runtime(self, infer_state: InferStateInfo) -> None:
        config = self.layer_config
        self.flash_kda_fwd = load_flash_kda_fwd(config.prefill_kernel, dtype=config.dtype, ssm_dtype=config.ssm_dtype)
        # Automatic selection keeps FlashKDA to the step lengths it was validated on.
        self.max_flash_len = None
        if config.prefill_kernel == "auto":
            self.max_flash_len = 2048 if config.num_heads < 16 else 8192

    def build(self, infer_state: InferStateInfo) -> KDAPrefillAttState:
        config, req, cu = self.layer_config, infer_state.b_req_idx, infer_state.b1_cu_q_seq_len
        flash = self.flash_kda_fwd
        if flash is not None and self.max_flash_len is not None and infer_state.max_q_seq_len > self.max_flash_len:
            flash = None
        # Native state keeps a slot per speculative row; replay state keeps one per request.
        slots_per_request = 1 if config.replay else config.state_mtp_step + 1
        return KDAPrefillAttState(
            conv_idx=req,
            ssm_idx=req * slots_per_request,
            cu_seqlens_q=cu,
            cu_seqlens=cu.to(torch.int64).contiguous() if flash is not None else cu,
            b_ready_cache_len=infer_state.b_ready_cache_len,
            max_q_seq_len=infer_state.max_q_seq_len,
            flash_kda_fwd=flash,
        )

    def forward(
        self,
        att_state: KDAPrefillAttState,
        layer_inputs: KDALayerInputs,
        *,
        out: Optional[torch.Tensor],
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Return the chunk output, writing it into out when given; update the conv and SSM states in place."""
        from lightllm.common.attention.kernels.kda.causal_conv1d_qkv import causal_conv1d_qkv

        config, inputs = self.layer_config, layer_inputs
        flash = att_state.flash_kda_fwd is not None
        q, k, v = causal_conv1d_qkv(
            inputs.mixed_qkv,
            inputs.conv_weight,
            # Prefill uses only the kernel window; speculative columns follow it.
            inputs.conv_states[..., : config.conv_kernel_size - 1],
            att_state.cu_seqlens_q,
            att_state.conv_idx,
            att_state.b_ready_cache_len,
            att_state.max_q_seq_len,
            use_qk_l2norm=not flash,
        )
        return _prefill(config, att_state, inputs, q, k, v, qk_normalized=not flash, out=out)


class ReplayKDAPrefillBackend(TritonKDAPrefillBackend):
    """Prefill over replay state, which keeps raw q/k: the fused conv runs only for FlashKDA with width 4."""

    name = "replay_kda_prefill"
    phase = "prefill"
    replay = True

    def forward(
        self,
        att_state: KDAPrefillAttState,
        layer_inputs: KDALayerInputs,
        *,
        out: Optional[torch.Tensor],
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Like native prefill, but replay state keeps the raw q/k, so they are not L2-normalized."""
        config, inputs = self.layer_config, layer_inputs
        # Prefill uses only the kernel window; speculative columns follow it.
        conv_states = inputs.conv_states[..., : config.conv_kernel_size - 1]
        if att_state.flash_kda_fwd is not None and config.conv_kernel_size == 4:
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
                x.view(1, -1, config.num_heads, config.head_dim)
                for x in mixed_qkv.split(config.num_heads * config.head_dim, -1)
            )
        return _prefill(config, att_state, inputs, q, k, v, qk_normalized=False, out=out)


class TritonKDADecodeBackend(AttentionBackend[KDALayerConfig, GDNDecodeAttState, KDALayerInputs]):
    name = "triton_kda_decode"
    phase = "decode"
    replay = False

    @classmethod
    def _check_supported(cls, layer_config: KDALayerConfig) -> None:
        _check_state(cls, layer_config)

    @classmethod
    def group_key(cls, layer_config: KDALayerConfig):
        return layer_config.replay

    def build(self, infer_state: InferStateInfo) -> GDNDecodeAttState:
        return build_recurrent_decode_state(self, infer_state)

    def forward(
        self,
        att_state: GDNDecodeAttState,
        layer_inputs: KDALayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out [rows, num_heads, head_dim]; update the conv and SSM states in place."""
        from lightllm.common.attention.kernels.gdn.third_party_fla.ops.kda_decode import fused_recurrent_kda

        config, inputs = self.layer_config, layer_inputs
        hidden = config.num_heads * config.head_dim
        q, k, v = _conv_update(inputs, att_state).split(hidden, dim=-1)
        shape = (1, -1) if att_state.draft_step > 0 else (-1, 1)
        fused_recurrent_kda(
            q=q.view(*shape, config.num_heads, config.head_dim),
            k=k.view(*shape, config.num_heads, config.head_dim),
            v=v.view(*shape, config.num_heads, config.head_dim),
            raw_gate=inputs.raw_gate.view(*shape, hidden),
            raw_beta=inputs.raw_beta.view(*shape, config.num_heads),
            a_log=inputs.A_log,
            gate_bias=inputs.dt_bias,
            initial_state=inputs.ssm_states,
            lower_bound=config.lower_bound,
            inplace_final_state=True,
            ssm_state_indices=att_state.ssm_idx,
            cu_seqlens=att_state.cu_seqlens,
            num_accepted_tokens=att_state.num_accepted_tokens,
            out=out.view(*shape, config.num_heads, config.head_dim),
        )
        return out


class ReplayKDADecodeBackend(TritonKDADecodeBackend):
    name = "replay_kda_decode"
    phase = "decode"
    replay = True

    def forward(
        self,
        att_state: GDNDecodeAttState,
        layer_inputs: KDALayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out [rows, num_heads, head_dim]; update the conv state and append history.

        Every layer reads the cursors of the step's start; the last layer's call advances them.
        """
        from lightllm.common.attention.kernels.kda.replay_kda import replay

        config, inputs = self.layer_config, layer_inputs
        hidden = config.num_heads * config.head_dim
        q, k, v = (
            x.view(-1, config.num_heads, config.head_dim) for x in _conv_update(inputs, att_state).split(hidden, -1)
        )
        state = inputs.replay_state
        layer = state.storage_layer(inputs.layer_num)
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
            b=inputs.raw_beta.reshape(-1, config.num_heads),
            a_log=inputs.A_log,
            bias=inputs.dt_bias,
            reqs=att_state.conv_idx,
            cu_seqlens=att_state.cu_seqlens,
            state_verify_width=state.verify_width,
            state_compute_mode=state.compute_mode,
            state_gate_lower_bound=state.gate_lower_bound,
            out=out.view(v.shape),
        )
        state.finish_decode_token(inputs.layer_num, att_state.conv_idx, att_state.cu_seqlens)
        return out
