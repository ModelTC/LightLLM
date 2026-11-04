"""Dense attention implementations consuming bound tensors and batch metadata."""

from dataclasses import dataclass, fields, replace
from typing import Callable

import torch

from lightllm.core.contracts.provider import AttentionOffer, ProbeResult
from lightllm.core.compute.attention.dense.spec import DenseAttentionSpec
from lightllm.core.compute.attention.dense.inputs import query_tensor
from lightllm.core.contracts.state import StateView


def decode_blocks(batch_size):
    return 128 if batch_size <= 16 else 64 if batch_size <= 64 else 32


def scratch_elements(max_batch_size, heads):
    # The largest batch does not necessarily have the largest split workspace.
    return max(min(max_batch_size, n) * decode_blocks(min(max_batch_size, n)) for n in (16, 64, max_batch_size)) * heads


@dataclass(frozen=True)
class DenseStaticParams:
    spec: DenseAttentionSpec
    alloc_tensor: Callable
    sinks: torch.Tensor | None = None

    @classmethod
    def from_weights(cls, spec, weight, *, device):
        return cls(spec, device.allocate_tensor, weight.attn_sinks.weight if spec.attention_sink else None)

    def validate_sinks(self, device):
        if not self.spec.attention_sink:
            if self.sinks is not None:
                raise ValueError("attention sink was not declared by the model")
        elif (
            self.sinks is None
            or self.sinks.shape != (self.spec.q_heads,)
            or self.sinks.dtype != torch.bfloat16
            or self.sinks.device != device
            or not self.sinks.is_contiguous()
        ):
            raise ValueError("FA3 attention requires one bound BF16 sink per query head")


@dataclass
class DenseBatchMetadata:
    batch_size: int
    max_q_len: int
    max_kv_len: int
    microbatch_index: int
    req_to_tokens: torch.Tensor
    req_indices: torch.Tensor
    seq_lengths: torch.Tensor
    q_starts: torch.Tensor | None
    ready_lengths: torch.Tensor | None
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    page_table: torch.Tensor | None = None
    total_kv_tokens: int | None = None
    total_q_tokens: int | None = None
    hold_request_id: int | None = None
    shared_groups: torch.Tensor | None = None
    mtp_indices: torch.Tensor | None = None
    host_seq_lengths: torch.Tensor | None = None
    host_cu_q: torch.Tensor | None = None
    host_cu_k: torch.Tensor | None = None
    host_ready: object = None

    def copy_for_graph(self, new):
        if self.batch_size != new.batch_size or self.microbatch_index != new.microbatch_index:
            raise ValueError("decode Graph metadata must retain its batch size and execution lane")
        if new.host_ready is not None:
            new.host_ready.synchronize()
        for field in fields(self):
            old, value = getattr(self, field.name), getattr(new, field.name)
            if isinstance(value, torch.Tensor):
                if old is None or old.shape != value.shape or old.dtype != value.dtype:
                    raise ValueError(f"Graph metadata shape/dtype changed: {field.name}")
                if old.data_ptr() != value.data_ptr():
                    old.copy_(value, non_blocking=True)


def group_verify_batch(batch, max_tokens):
    """Coalesce consecutive query rows while keeping Graph metadata shapes fixed."""
    from lightllm.common.basemodel.triton_kernel.mtp_utils import build_mtp_shared_group_markers
    from lightllm.common.basemodel.triton_kernel.fa3_utils import build_dynamic_spec_fa3_decode_params
    from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor

    if batch.hold_request_id is None:
        raise ValueError("verification requires an explicit padding request identity")
    groups = build_mtp_shared_group_markers(batch.req_indices, batch.hold_request_id)
    q_lengths, kv_lengths, requests, lengths = build_dynamic_spec_fa3_decode_params(
        batch.req_indices, batch.seq_lengths, groups, batch.batch_size, batch.hold_request_id
    )
    cu_q, cu_k = gen_cumsum_pad0_tensor(q_lengths, kv_lengths)
    return replace(
        batch,
        req_indices=requests,
        seq_lengths=lengths,
        cu_q=cu_q.int(),
        cu_k=cu_k.int(),
        q_starts=cu_q[:-1].int(),
        ready_lengths=lengths - q_lengths,
        max_q_len=max_tokens,
    )


class BoundAttention:
    def __init__(self, offer, state_view, static_params, workspace):
        self.name, self.phase = offer.backend, offer.phase
        self.implementation = self.name.split(".")[0]
        self.view, self.spec = state_view, static_params.spec
        self.alloc_tensor = static_params.alloc_tensor
        self.sinks = static_params.sinks
        self.k, self.v = state_view.fields[:2]
        self.page_size = state_view.format.page_size
        self.limits = offer.execution_limits
        self.transfer_requires_compute_sync = offer.transfer_requires_compute_sync
        self.workspace = workspace

    @property
    def metadata_key(self):
        return self.name

    def _lane_workspace(self, index):
        if not 0 <= index < self.limits.microbatch_count:
            raise ValueError("microbatch exceeds reserved execution lanes")
        if self.workspace is None:
            return None
        size = self.workspace.numel() // self.limits.microbatch_count
        return self.workspace[index * size : (index + 1) * size]

    def create_metadata(self, batch):
        if batch.batch_size > self.limits.max_batch_size or batch.max_kv_len > self.limits.max_seq_len:
            raise ValueError("batch exceeds the planned attention capacity")
        scratch = self._lane_workspace(batch.microbatch_index)
        if self.phase == "verify":
            if self.implementation == "fa3":
                batch = group_verify_batch(batch, self.limits.max_verify_tokens)
            elif isinstance(self.spec, DenseAttentionSpec) and self.spec.window == (-1, -1):
                from lightllm.common.basemodel.triton_kernel.mtp_utils import build_mtp_shared_group_markers

                if batch.hold_request_id is None:
                    raise ValueError("verification requires an explicit padding request identity")
                batch = replace(
                    batch, shared_groups=build_mtp_shared_group_markers(batch.req_indices, batch.hold_request_id)
                )
        metadata = replace(batch, cu_q=batch.cu_q.int(), cu_k=batch.cu_k.int())
        if self.implementation == "fa3":
            from lightllm.common.basemodel.triton_kernel.fa3_utils import page_table_copy

            page_size = self.page_size
            width = (self.limits.max_seq_len + page_size - 1) // page_size
            metadata.page_table = scratch.view(torch.int32)[: batch.batch_size * width].view(batch.batch_size, width)
            active_width = (batch.max_kv_len + page_size - 1) // page_size
            page_table_copy(
                metadata.page_table[:, :active_width], metadata.req_to_tokens, metadata.req_indices, page_size
            )
        return metadata

    def prepare_batch(self, batch):
        return batch.execution_metadata[self.metadata_key]

    def run(self, inputs, metadata, out=None):
        inputs = query_tensor(inputs)
        if self.implementation == "fa3":
            result = self._fa3(inputs, metadata)
            if out is not None:
                out.copy_(result)
                return out
            return result
        out = self.alloc_tensor(inputs.shape, inputs.dtype) if out is None else out
        if self.phase == "prefill":
            from lightllm.common.basemodel.triton_kernel.att.prefill_att.context_flashattention_nopad import (
                context_attention_fwd,
            )

            context_attention_fwd(
                inputs,
                self.k,
                self.v,
                out,
                metadata.req_indices,
                metadata.q_starts,
                metadata.seq_lengths,
                metadata.ready_lengths,
                metadata.max_q_len,
                metadata.req_to_tokens,
                sliding_window=self.spec.window,
            )
        else:
            self._triton_decode(inputs, metadata, out)
        return out

    def _triton_decode(self, q, metadata, out):
        from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.flash_decoding import (
            gqa_flash_decoding_stage1 as stage1,
            gqa_flash_decoding_stage2 as stage2,
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
        if self.phase == "verify" and self.spec.window == (-1, -1):
            from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.mtp_diverse import (
                mtp_diverse_stage1_single_token,
                mtp_diverse_stage2_single_token,
            )
            from lightllm.utils.envs_utils import get_diverse_max_batch_shared_group_size

            block_n = mtp_diverse_stage1_single_token(
                q=q,
                k=self.k,
                v=self.v,
                Req_to_tokens=metadata.req_to_tokens,
                B_req_idx=metadata.req_indices,
                b_seq_len=metadata.seq_lengths,
                b_mark_shared_group=metadata.shared_groups,
                max_kv_len=self.limits.max_seq_len,
                mid_out=mid,
                mid_out_logsumexp=lse,
                block_batch=get_diverse_max_batch_shared_group_size(),
            )
            mtp_diverse_stage2_single_token(
                mid_out=mid,
                mid_out_logsumexp=lse,
                B_Seqlen=metadata.seq_lengths,
                out=out,
                block_n=block_n,
                max_kv_len=self.limits.max_seq_len,
            )
            return
        stage1.flash_decode_stage1(
            q=q,
            k=self.k,
            v=self.v,
            Req_to_tokens=metadata.req_to_tokens,
            B_req_idx=metadata.req_indices,
            B_Seqlen=metadata.seq_lengths,
            max_len_in_batch=metadata.max_kv_len,
            mid_out=mid,
            mid_out_logsumexp=lse,
            block_seq=256,
            sliding_window=self.spec.window,
        )
        stage2.flash_decode_stage2(
            mid_out=mid,
            mid_out_logexpsum=lse,
            B_Seqlen=metadata.seq_lengths,
            out=out,
            block_seq=256,
            sliding_window=self.spec.window,
        )

    def _fa3(self, q, metadata):
        from lightllm.utils.sgl_utils import (
            flash_attn_with_kvcache,
            flash_attn_with_kvcache_autotune,
        )

        function = flash_attn_with_kvcache if self.phase == "prefill" else flash_attn_with_kvcache_autotune
        page_size = self.page_size
        kwargs = {} if self.phase == "prefill" else {"max_seqlen_k": metadata.max_kv_len}
        return function(
            q=q,
            k_cache=self.k.view(-1, page_size, self.spec.kv_heads, self.spec.head_dim),
            v_cache=self.v.view(-1, page_size, self.spec.kv_heads, self.spec.head_dim),
            page_table=metadata.page_table,
            cache_seqlens=metadata.seq_lengths,
            cu_seqlens_q=metadata.cu_q,
            cu_seqlens_k_new=metadata.cu_k if self.phase == "prefill" else None,
            max_seqlen_q=metadata.max_q_len if self.phase != "decode" else 1,
            softmax_scale=self.spec.scale,
            causal=self.spec.causal,
            window_size=self.spec.window,
            softcap=self.spec.softcap,
            k_descale=None,
            v_descale=None,
            return_softmax_lse=False,
            sinks=self.sinks,
            **kwargs,
        )


class DenseAttentionBackend:
    def __init__(self, name: str, phase: str):
        if name not in ("triton", "fa3") or phase not in ("prefill", "decode", "verify"):
            raise ValueError("Dense supports Triton/FA3 prefill, decode and verify")
        self.implementation, self.phase = name, phase
        self.name = f"{name}.{phase}"

    def supports_device(self, caps):
        return caps.is_nvidia

    def probe(self, attention_spec, format_spec, device_caps, limits):
        if not isinstance(attention_spec, DenseAttentionSpec):
            return ProbeResult(reason="requires Dense attention semantics")
        if attention_spec.attention_sink and self.implementation != "fa3":
            return ProbeResult(reason="attention sinks require a supporting backend")
        if not self.supports_device(device_caps):
            return ProbeResult(reason=f"device is not supported by {self.implementation}")
        if self.implementation == "fa3" and (device_caps.capability is None or device_caps.capability[0] != 9):
            return ProbeResult(reason="FA3 requires Hopper capability")
        if self.implementation == "fa3":
            from lightllm.utils.sgl_utils import flash_attn_with_kvcache

            if flash_attn_with_kvcache is None:
                return ProbeResult(reason="FA3 kernel is not installed")
        if (
            format_spec.encoding != "plain"
            or format_spec.layout != "token_head_dim"
            or format_spec.fields != ("k", "v")
            or format_spec.dtype not in (torch.float16, torch.bfloat16)
            or format_spec.version != 1
            or attention_spec.dtype not in (None, format_spec.dtype)
        ):
            return ProbeResult(reason="requires plain FP16/BF16 token/head/dim K and V")
        if (
            (not attention_spec.causal and self.implementation != "fa3")
            or (
                attention_spec.window != (-1, -1)
                and not (
                    attention_spec.window[0] >= 0
                    and (
                        attention_spec.window[1] == 0
                        or (
                            self.implementation == "fa3" and not attention_spec.causal and attention_spec.window[1] >= 0
                        )
                    )
                )
            )
            or attention_spec.softcap != 0
            or attention_spec.scale != attention_spec.head_dim ** -0.5
        ):
            return ProbeResult(
                reason="requires default scale and no softcap; non-causal attention and right windows require FA3"
            )
        if attention_spec.head_dim not in (16, 32, 64, 128, 256):
            return ProbeResult(reason="unsupported Dense head dimension")
        if self.implementation == "fa3" and attention_spec.head_dim < 64:
            return ProbeResult(reason="FA3 requires head dimension >= 64")
        size = 0
        if self.implementation == "fa3":
            size = (
                limits.max_batch_size * ((limits.max_seq_len + format_spec.page_size - 1) // format_spec.page_size) * 4
            )
        elif self.phase != "prefill":
            size = scratch_elements(limits.max_batch_size, attention_spec.q_heads) * (
                attention_spec.head_dim * format_spec.dtype.itemsize + 4
            )
        return ProbeResult(
            offers=(
                AttentionOffer(
                    self.name,
                    self.phase,
                    format_spec,
                    size * limits.microbatch_count,
                    workspace_group="serial_layers",
                    execution_limits=limits,
                    transfer_requires_compute_sync=self.implementation == "fa3",
                ),
            )
        )

    def bind(self, offer, state_view: StateView, static_params: DenseStaticParams, workspace):
        if offer.backend != self.name or offer.phase != self.phase or offer.format != state_view.format:
            raise ValueError("offer does not match the backend and state view")
        k, v = state_view.fields
        spec = static_params.spec
        static_params.validate_sinks(k.device)
        if k.shape != v.shape or k.ndim != 3 or k.shape[1:] != (spec.kv_heads, spec.head_dim):
            raise ValueError("state view shape disagrees with the layer's attention spec")
        if k.dtype != state_view.format.dtype or v.dtype != k.dtype or k.device != v.device:
            raise ValueError("state view dtype/device disagrees with its format")
        if k.stride() != v.stride() or k.stride(-1) != 1:
            raise ValueError("Dense K/V must have matching strides and contiguous head dimensions")
        if k.shape[0] % state_view.format.page_size:
            raise ValueError("state capacity must contain complete pages")
        if offer.workspace_bytes and (
            workspace is None
            or workspace.dtype != torch.uint8
            or workspace.numel() < offer.workspace_bytes
            or workspace.device != k.device
            or not workspace.is_contiguous()
        ):
            raise ValueError("workspace does not satisfy the selected offer")
        return BoundAttention(offer, state_view, static_params, workspace)


from lightllm.core.formats.kv.dense import DenseKVWriter
