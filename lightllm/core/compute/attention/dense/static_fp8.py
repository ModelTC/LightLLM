"""FA3 FP8 execution with calibration and native cache views bound at startup."""

import torch

from lightllm.core.compute.attention.dense.backend import BoundAttention
from lightllm.core.contracts.provider import AttentionOffer, ProbeResult
from lightllm.core.compute.attention.dense.spec import DenseAttentionSpec


class BoundStaticFP8Attention(BoundAttention):
    def __init__(self, offer, state_view, static_params, workspace):
        super().__init__(offer, state_view, static_params, workspace)
        # FA3 FP8 uses single-token inference pages, independently of allocator pages.
        self.page_size = 1
        scales, self.q_scale = state_view.fields[2:]
        width = 1 if state_view.format.encoding == "fp8_per_tensor" else self.spec.kv_heads
        self.k_scale = scales[:width].expand(self.spec.kv_heads)
        self.v_scale = scales[width:].expand(self.spec.kv_heads)

    def run(self, inputs, metadata, out=None):
        from lightllm.common.basemodel.triton_kernel.quantization.q_per_head_fp8_quant import (
            q_per_head_fp8_quant,
            q_per_head_static_fp8_quant,
        )
        from lightllm.utils.sgl_utils import flash_attn_with_kvcache

        if inputs.dtype != self.spec.dtype:
            raise ValueError("query dtype differs from the planned execution dtype")
        grouped = inputs.reshape(inputs.shape[0], self.spec.kv_heads, -1)
        if self.q_scale is None:
            q, q_scale = q_per_head_fp8_quant(grouped, metadata.cu_q[1:] - metadata.cu_q[:-1], metadata.cu_q)
        else:
            q = q_per_head_static_fp8_quant(grouped, self.q_scale)
            q_scale = self.q_scale[None, :].expand(metadata.batch_size, -1)
        result = flash_attn_with_kvcache(
            q=q.reshape(inputs.shape),
            k_cache=self.k[:, None],
            v_cache=self.v[:, None],
            page_table=metadata.page_table,
            cache_seqlens=metadata.seq_lengths,
            cu_seqlens_q=metadata.cu_q,
            cu_seqlens_k_new=metadata.cu_k if self.phase == "prefill" else None,
            max_seqlen_q=metadata.max_q_len if self.phase != "decode" else 1,
            softmax_scale=self.spec.scale,
            causal=self.spec.causal,
            window_size=self.spec.window,
            softcap=self.spec.softcap,
            q_descale=q_scale,
            k_descale=self.k_scale[None, :].expand(metadata.batch_size, -1),
            v_descale=self.v_scale[None, :].expand(metadata.batch_size, -1),
            return_softmax_lse=False,
            sinks=self.sinks,
        )
        if out is not None:
            out.copy_(result)
            return out
        return result.to(inputs.dtype)


class StaticFP8AttentionBackend:
    implementation = "fa3"

    def __init__(self, phase):
        if phase not in ("prefill", "decode", "verify"):
            raise ValueError("FP8 attention supports prefill and decode")
        self.phase, self.name = phase, f"fa3.fp8.{phase}"

    def probe(self, attention_spec, format_spec, device_caps, limits):
        if not isinstance(attention_spec, DenseAttentionSpec):
            return ProbeResult(reason="requires Dense attention semantics")
        if device_caps.device_type != "cuda" or device_caps.capability is None or device_caps.capability[0] != 9:
            return ProbeResult(reason="FP8 FA3 requires Hopper")
        from lightllm.utils.sgl_utils import flash_attn_with_kvcache

        if flash_attn_with_kvcache is None:
            return ProbeResult(reason="FA3 kernel is not installed")
        spec, format = attention_spec, format_spec
        if (
            format.encoding not in ("fp8_per_head", "fp8_per_tensor")
            or format.dtype != torch.float8_e4m3fn
            or format.scale_dtype != torch.float32
            or format.fields != ("k", "v", "kv_scale", "q_scale")
            or format.layout != "token_head_dim"
            or format.version != 1
        ):
            return ProbeResult(reason="requires static E4M3 FP8 with explicit calibration fields")
        if format.encoding == "fp8_per_head" and format.calibration is None:
            return ProbeResult(reason="per-head FP8 requires a calibration identity")
        if (
            spec.dtype not in (torch.float16, torch.bfloat16)
            or spec.head_dim not in (64, 128, 256)
            or spec.softcap
            or spec.scale != spec.head_dim ** -0.5
            or (spec.window != (-1, -1) and not (spec.window[0] >= 0 and spec.window[1] == 0))
        ):
            return ProbeResult(reason="requires FP16/BF16 Dense with supported head size and optional window")
        return ProbeResult(
            offers=(
                AttentionOffer(
                    self.name,
                    self.phase,
                    format,
                    limits.max_batch_size * limits.max_seq_len * 4 * limits.microbatch_count,
                    workspace_group="serial_layers",
                    execution_limits=limits,
                    transfer_requires_compute_sync=True,
                ),
            )
        )

    def bind(self, offer, state_view, static_params, workspace):
        if offer.backend != self.name or offer.phase != self.phase or offer.format != state_view.format:
            raise ValueError("offer does not match the backend and state view")
        k, v, scales, q_scale = state_view.fields
        spec = static_params.spec
        static_params.validate_sinks(k.device)
        if k.ndim != 3 or k.shape != v.shape or k.shape[1:] != (spec.kv_heads, spec.head_dim):
            raise ValueError("FP8 payload shape differs from attention semantics")
        if any(t.dtype != torch.float8_e4m3fn or t.device != k.device or not t.is_contiguous() for t in (k, v)):
            raise ValueError("FP8 payload requires contiguous E4M3 tensors on the same device")
        width = 1 if state_view.format.encoding == "fp8_per_tensor" else spec.kv_heads
        if scales is None:
            raise ValueError("KV calibration is required")
        for tensor, size in ((scales, 2 * width), (q_scale, spec.kv_heads)):
            if tensor is None:
                continue
            if (
                tensor.shape != (size,)
                or tensor.dtype != torch.float32
                or tensor.device != k.device
                or not tensor.is_contiguous()
                or not bool((torch.isfinite(tensor) & (tensor > 0)).all())
            ):
                raise ValueError("calibration requires finite positive FP32 scales matching the bound heads")
        if (
            workspace is None
            or workspace.dtype != torch.uint8
            or workspace.numel() < offer.workspace_bytes
            or workspace.device != k.device
            or not workspace.is_contiguous()
        ):
            raise ValueError("workspace does not satisfy the selected offer")
        return BoundStaticFP8Attention(offer, state_view, static_params, workspace)


from lightllm.core.formats.kv.static_fp8 import StaticFP8KVWriter
