"""KDA implementations: chunked prefill (Triton, or FlashKDA for short steps) and decode over native or
replay recurrent state. Decode rows group into requests with the GDN decode step builder."""

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from lightllm.common.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionBuildResources,
    TensorAllocator,
)
from ..gdn.triton import GDNDecodeAttState, GDNDecodeBuildContext
from . import KDALayerConfig, KDALayerInputs
from .flashkda import load_flash_kda_fwd


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


class KDAPrefillBuildContext:
    def __init__(
        self,
        *,
        ssm_slots_per_request: int,
        flash_kda_fwd: Optional[Callable] = None,
        max_flash_len: Optional[int] = None,
    ) -> None:
        """flash_kda_fwd is FlashKDA's kernel when it may run; steps whose longest request exceeds
        max_flash_len (None: no limit) use Triton instead."""
        self.ssm_slots_per_request = ssm_slots_per_request
        self.flash_kda_fwd = flash_kda_fwd
        self.max_flash_len = max_flash_len

    def prepare(self, batch: AttentionMetadata) -> KDAPrefillAttState:
        req = batch.b_req_idx
        flash = self.flash_kda_fwd
        if flash is not None and self.max_flash_len is not None and batch.max_q_seq_len > self.max_flash_len:
            flash = None
        cu = batch.cu_q_seq_len
        return KDAPrefillAttState(
            conv_idx=req,
            ssm_idx=req * self.ssm_slots_per_request,
            cu_seqlens_q=cu,
            cu_seqlens=cu.to(torch.int64).contiguous() if flash is not None else cu,
            b_ready_cache_len=batch.b_ready_cache_len,
            max_q_seq_len=batch.max_q_seq_len,
            flash_kda_fwd=flash,
        )


def _conv_states(cfg: KDALayerConfig, inputs: KDALayerInputs):
    # Prefill uses only the kernel window; speculative columns follow it.
    return inputs.conv_states[..., : cfg.conv_kernel_size - 1]


def _prefill(cfg: KDALayerConfig, att_state: KDAPrefillAttState, inputs: KDALayerInputs, q, k, v, qk_normalized, out):
    from lightllm.common.attention.kernels.kda.kda_prefill import kda_prefill

    output = kda_prefill(
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
    )
    out.copy_(output.view(out.shape))
    return out


class TritonKDAPrefillBackend(AttentionBackend[KDALayerConfig, KDAPrefillAttState, KDALayerInputs]):
    """Prefill over native state: the fused conv emits q/k/v (L2-normalized unless FlashKDA runs)."""

    def __init__(self, layer_config: KDALayerConfig) -> None:
        super().__init__(layer_config)
        _check_state(layer_config, replay=False)
        _check_prefill_kernel(layer_config)

    def forward(self, att_state, layer_inputs, *, out, alloc_func: TensorAllocator = torch.empty):
        from lightllm.common.attention.kernels.kda.causal_conv1d_qkv import causal_conv1d_qkv

        cfg, inputs = self.layer_config, layer_inputs
        flash = att_state.flash_kda_fwd is not None
        q, k, v = causal_conv1d_qkv(
            inputs.mixed_qkv,
            inputs.conv_weight,
            _conv_states(cfg, inputs),
            att_state.cu_seqlens_q,
            att_state.conv_idx,
            att_state.b_ready_cache_len,
            att_state.max_q_seq_len,
            use_qk_l2norm=not flash,
        )
        return _prefill(cfg, att_state, inputs, q, k, v, qk_normalized=not flash, out=out)

    @staticmethod
    def get_name() -> str:
        return "triton_kda_prefill"

    build_context_type = KDAPrefillBuildContext

    @classmethod
    def build(cls, metadata: AttentionMetadata, context: KDAPrefillBuildContext) -> KDAPrefillAttState:
        return context.prepare(metadata)

    @classmethod
    def create_build_context(cls, layer_config: KDALayerConfig, ctx: AttentionBuildResources) -> KDAPrefillBuildContext:
        return _create_prefill_step_builder(layer_config)

    @classmethod
    def build_context_key(cls, layer_config: KDALayerConfig):
        return _prefill_build_context_key(layer_config)


class ReplayKDAPrefillBackend(AttentionBackend[KDALayerConfig, KDAPrefillAttState, KDALayerInputs]):
    """Prefill over replay state, which keeps raw q/k: the fused conv runs only for FlashKDA with width 4."""

    def __init__(self, layer_config: KDALayerConfig) -> None:
        super().__init__(layer_config)
        _check_state(layer_config, replay=True)
        _check_prefill_kernel(layer_config)

    def forward(self, att_state, layer_inputs, *, out, alloc_func: TensorAllocator = torch.empty):
        cfg, inputs = self.layer_config, layer_inputs
        conv_states = _conv_states(cfg, inputs)
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
        return _prefill(cfg, att_state, inputs, q, k, v, qk_normalized=False, out=out)

    @staticmethod
    def get_name() -> str:
        return "replay_kda_prefill"

    build_context_type = KDAPrefillBuildContext

    @classmethod
    def build(cls, metadata: AttentionMetadata, context: KDAPrefillBuildContext) -> KDAPrefillAttState:
        return context.prepare(metadata)

    @classmethod
    def create_build_context(cls, layer_config: KDALayerConfig, ctx: AttentionBuildResources) -> KDAPrefillBuildContext:
        return _create_prefill_step_builder(layer_config)

    @classmethod
    def build_context_key(cls, layer_config: KDALayerConfig):
        return _prefill_build_context_key(layer_config)


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


class TritonKDADecodeBackend(AttentionBackend[KDALayerConfig, GDNDecodeAttState, KDALayerInputs]):
    def __init__(self, layer_config: KDALayerConfig) -> None:
        super().__init__(layer_config)
        _check_state(layer_config, replay=False)

    def forward(self, att_state, layer_inputs, *, out, alloc_func: TensorAllocator = torch.empty):
        """Fill and return out [rows, num_heads, head_dim]; update the conv and SSM states in place."""
        from lightllm.common.attention.kernels.gdn.third_party_fla.ops.kda_decode import fused_recurrent_kda

        cfg, inputs = self.layer_config, layer_inputs
        hidden = cfg.num_heads * cfg.head_dim
        q, k, v = _conv_update(inputs, att_state).split(hidden, dim=-1)
        shape = (1, -1) if att_state.draft_step > 0 else (-1, 1)
        output, _ = fused_recurrent_kda(
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
        )
        out.copy_(output.view(out.shape))
        return out

    @staticmethod
    def get_name() -> str:
        return "triton_kda_decode"

    build_context_type = GDNDecodeBuildContext

    @classmethod
    def build(cls, metadata: AttentionMetadata, context: GDNDecodeBuildContext) -> GDNDecodeAttState:
        return context.prepare(metadata)

    @classmethod
    def create_build_context(cls, layer_config: KDALayerConfig, ctx: AttentionBuildResources) -> GDNDecodeBuildContext:
        return GDNDecodeBuildContext(replay=layer_config.replay)

    @classmethod
    def build_context_key(cls, layer_config: KDALayerConfig):
        return layer_config.replay


class ReplayKDADecodeBackend(AttentionBackend[KDALayerConfig, GDNDecodeAttState, KDALayerInputs]):
    def __init__(self, layer_config: KDALayerConfig) -> None:
        super().__init__(layer_config)
        _check_state(layer_config, replay=True)

    def forward(self, att_state, layer_inputs, *, out, alloc_func: TensorAllocator = torch.empty):
        """Fill and return out [rows, num_heads, head_dim]; update the conv state and append history.

        Every layer reads the cursors of the step's start; the last layer's call advances them.
        """
        from lightllm.common.attention.kernels.kda.replay_kda import replay

        cfg, inputs = self.layer_config, layer_inputs
        hidden = cfg.num_heads * cfg.head_dim
        q, k, v = (x.view(-1, cfg.num_heads, cfg.head_dim) for x in _conv_update(inputs, att_state).split(hidden, -1))
        state = inputs.replay_state
        layer = state.storage_layer(inputs.layer_num)
        output = replay(
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
        )
        out.copy_(output.view(out.shape))
        state.finish_decode_token(inputs.layer_num, att_state.conv_idx, att_state.cu_seqlens)
        return out

    @staticmethod
    def get_name() -> str:
        return "replay_kda_decode"

    build_context_type = GDNDecodeBuildContext

    @classmethod
    def build(cls, metadata: AttentionMetadata, context: GDNDecodeBuildContext) -> GDNDecodeAttState:
        return context.prepare(metadata)

    @classmethod
    def create_build_context(cls, layer_config: KDALayerConfig, ctx: AttentionBuildResources) -> GDNDecodeBuildContext:
        return GDNDecodeBuildContext(replay=layer_config.replay)

    @classmethod
    def build_context_key(cls, layer_config: KDALayerConfig):
        return layer_config.replay


def _check_state(layer_config: KDALayerConfig, *, replay: bool) -> None:
    if layer_config.replay != replay:
        state = "replay" if layer_config.replay else "native"
        raise NotImplementedError(f"this KDA implementation does not run over {state} state")


def _check_prefill_kernel(layer_config: KDALayerConfig) -> None:
    # A requested FlashKDA must load when the model loads, not on the first prefill step.
    if layer_config.prefill_kernel == "flashkda":
        from .flashkda import import_flash_kda_fwd

        try:
            import_flash_kda_fwd()
        except ImportError as e:
            raise NotImplementedError(str(e)) from e


def _create_prefill_step_builder(layer_config: KDALayerConfig) -> KDAPrefillBuildContext:
    flash_kda_fwd = load_flash_kda_fwd(
        layer_config.prefill_kernel, dtype=layer_config.dtype, ssm_dtype=layer_config.ssm_dtype
    )
    # Automatic selection keeps FlashKDA to the step lengths it was validated on.
    max_flash_len = None
    if layer_config.prefill_kernel == "auto":
        max_flash_len = 2048 if layer_config.num_heads < 16 else 8192
    return KDAPrefillBuildContext(
        ssm_slots_per_request=1 if layer_config.replay else layer_config.state_mtp_step + 1,
        flash_kda_fwd=flash_kda_fwd,
        max_flash_len=max_flash_len,
    )


def _prefill_build_context_key(layer_config: KDALayerConfig):
    return (
        layer_config.replay,
        layer_config.state_mtp_step,
        layer_config.prefill_kernel,
        layer_config.num_heads,
        layer_config.dtype,
        layer_config.ssm_dtype,
    )
