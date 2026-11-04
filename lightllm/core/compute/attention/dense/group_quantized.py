"""Grouped INT4/INT8 attention consumes payload and scale views bound at startup."""

import torch

from lightllm.core.compute.attention.dense.inputs import query_tensor

from lightllm.core.compute.attention.dense.backend import BoundAttention, decode_blocks, scratch_elements
from lightllm.core.contracts.provider import AttentionOffer, ProbeResult
from lightllm.core.compute.attention.dense.spec import DenseAttentionSpec


class BoundGroupQuantizedAttention(BoundAttention):
    def __init__(self, offer, state_view, static_params, workspace):
        super().__init__(offer, state_view, static_params, workspace)
        self.k_scale, self.v_scale = state_view.fields[2:]
        self.bits = 4 if state_view.format.encoding == "int4" else 8
        self.dtype = self.spec.dtype or state_view.format.scale_dtype

    def run(self, inputs, metadata, out=None):
        inputs = query_tensor(inputs)
        if inputs.dtype != self.dtype:
            raise ValueError("query dtype differs from the planned execution dtype")
        out = self.alloc_tensor(inputs.shape, inputs.dtype) if out is None else out
        if self.phase == "prefill" or self.spec.window != (-1, -1):
            from lightllm.common.basemodel.triton_kernel.att.prefill_att.context_flashattention_nopad import (
                context_attention_fwd,
            )

            context_attention_fwd(
                inputs,
                self.k,
                self.v,
                out,
                metadata.req_indices,
                metadata.q_starts if self.phase == "prefill" else metadata.cu_q[:-1],
                metadata.seq_lengths,
                metadata.ready_lengths if self.phase == "prefill" else metadata.seq_lengths - 1,
                metadata.max_q_len if self.phase == "prefill" else 1,
                metadata.req_to_tokens,
                sliding_window=self.spec.window,
                k_scale=self.k_scale,
                v_scale=self.v_scale,
                kv_bits=self.bits,
                quant_group_size=self.view.format.group_size,
            )
        else:
            self._decode(inputs, metadata, out)
        return out

    def _decode(self, q, metadata, out):
        from lightllm.common.basemodel.triton_kernel.att.decode_att.int8kv.normal.int8kv_flash_decoding_stage2 import (
            flash_decode_stage2,
        )

        scratch = self._lane_workspace(metadata.microbatch_index)
        blocks = decode_blocks(metadata.batch_size)
        elements = metadata.batch_size * self.spec.q_heads * blocks
        value_bytes = elements * self.spec.head_dim * q.element_size()
        mid = (
            scratch[:value_bytes].view(q.dtype).view(metadata.batch_size, self.spec.q_heads, blocks, self.spec.head_dim)
        )
        lse = (
            scratch[value_bytes : value_bytes + elements * 4]
            .view(torch.float32)
            .view(metadata.batch_size, self.spec.q_heads, blocks)
        )
        args = dict(
            q=q,
            k=self.k,
            k_scale=self.k_scale,
            v=self.v,
            v_scale=self.v_scale,
            Req_to_tokens=metadata.req_to_tokens,
            B_req_idx=metadata.req_indices,
            mid_out=mid,
            mid_out_logsumexp=lse,
            block_seq=256,
        )
        if self.bits == 4:
            from lightllm.common.basemodel.triton_kernel.att.decode_att.int4kv.int4kv_flash_decoding_stage1 import (
                int4kv_flash_decode_stage1,
            )

            int4kv_flash_decode_stage1(
                **args,
                B_Seqlen=metadata.seq_lengths,
                max_kv_seq_len=metadata.max_kv_len,
            )
        else:
            from lightllm.common.basemodel.triton_kernel.att.decode_att.int8kv.normal import (
                int8kv_flash_decoding_stage1 as int8_stage,
            )

            int8_stage.flash_decode_stage1(
                **args,
                B_seq_len=metadata.seq_lengths,
                max_len_in_batch=metadata.max_kv_len,
            )
        flash_decode_stage2(mid, lse, metadata.seq_lengths, out, 256)


class GroupQuantizedAttentionBackend:
    implementation = "triton"

    def __init__(self, phase):
        if phase not in ("prefill", "decode", "verify"):
            raise ValueError("grouped attention supports prefill and decode")
        self.phase = phase
        self.name = f"triton.intkv.{phase}"

    def probe(self, attention_spec, format_spec, device_caps, limits):
        if not isinstance(attention_spec, DenseAttentionSpec):
            return ProbeResult(reason="requires Dense attention semantics")
        if attention_spec.attention_sink:
            return ProbeResult(reason="grouped attention does not implement attention sinks")
        if not device_caps.is_nvidia:
            return ProbeResult(reason="requires CUDA")
        spec, format = attention_spec, format_spec
        if (
            format.encoding not in ("int4", "int8")
            or format.dtype != torch.int8
            or format.fields != ("k", "v", "k_scale", "v_scale")
            or format.layout != "token_head_dim"
            or format.version != 1
            or format.scale_dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or format.group_size is None
            or spec.head_dim % format.group_size
        ):
            return ProbeResult(reason="requires grouped INT4/INT8 payload and complete floating scale fields")
        dtype = spec.dtype or format.scale_dtype
        if dtype not in (torch.float16, torch.bfloat16):
            return ProbeResult(reason="requires FP16/BF16 query dtype")
        if (
            not spec.causal
            or spec.softcap
            or spec.scale != spec.head_dim ** -0.5
            or (spec.window != (-1, -1) and not (spec.window[0] >= 0 and spec.window[1] == 0))
        ):
            return ProbeResult(reason="requires causal attention with default scale and optional left window")
        if spec.head_dim not in (16, 32, 64, 128, 256):
            return ProbeResult(reason="unsupported grouped attention head dimension")
        size = 0
        if self.phase != "prefill" and spec.window == (-1, -1):
            size = scratch_elements(limits.max_batch_size, spec.q_heads) * (spec.head_dim * dtype.itemsize + 4)
        return ProbeResult(
            offers=(
                AttentionOffer(
                    self.name,
                    self.phase,
                    format,
                    size * limits.microbatch_count,
                    workspace_group="serial_layers",
                    execution_limits=limits,
                ),
            )
        )

    def bind(self, offer, state_view, static_params, workspace):
        if offer.backend != self.name or offer.phase != self.phase or offer.format != state_view.format:
            raise ValueError("offer does not match the backend and state view")
        k, v, k_scale, v_scale = state_view.fields
        spec, format = static_params.spec, state_view.format
        packed_dim = spec.head_dim // (2 if format.encoding == "int4" else 1)
        if k.ndim != 3 or k.shape != v.shape or k.shape[1:] != (spec.kv_heads, packed_dim):
            raise ValueError("quantized payload shape disagrees with attention semantics")
        scale_shape = (k.shape[0], spec.kv_heads, spec.head_dim // format.group_size)
        if k_scale.shape != scale_shape or v_scale.shape != scale_shape:
            raise ValueError("scale shape disagrees with the quantization groups")
        if (
            k.dtype != torch.int8
            or v.dtype != k.dtype
            or k_scale.dtype != format.scale_dtype
            or v_scale.dtype != k_scale.dtype
        ):
            raise ValueError("payload or scale dtype differs from the selected format")
        if any(t.device != k.device or not t.is_contiguous() for t in state_view.fields):
            raise ValueError("grouped decode requires contiguous fields on the same device")
        if k.shape[0] % format.page_size:
            raise ValueError("state capacity must contain complete pages")
        if offer.workspace_bytes and (
            workspace is None
            or workspace.dtype != torch.uint8
            or workspace.numel() < offer.workspace_bytes
            or workspace.device != k.device
            or not workspace.is_contiguous()
        ):
            raise ValueError("workspace does not satisfy the selected offer")
        return BoundGroupQuantizedAttention(offer, state_view, static_params, workspace)


from lightllm.core.formats.kv.group_quantized import GroupQuantizedKVWriter
