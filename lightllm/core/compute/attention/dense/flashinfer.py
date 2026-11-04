"""FlashInfer's batch planning is explicit and separate from bound kernel execution."""

from dataclasses import dataclass, replace
from importlib.util import find_spec

import torch

from lightllm.core.compute.attention.dense.inputs import query_tensor

from lightllm.core.compute.attention.dense.backend import BoundAttention, group_verify_batch
from lightllm.core.contracts.provider import AttentionOffer, ProbeResult
from lightllm.core.compute.attention.dense.spec import DenseAttentionSpec

FLOAT_WORKSPACE_BYTES = 128 * 1024 ** 2
INT_WORKSPACE_BYTES = 8 * 1024 ** 2


@dataclass
class FlashInferBatchMetadata:
    owner: object
    batch: object
    indptr: torch.Tensor
    indices: torch.Tensor
    last_page_len: torch.Tensor
    wrapper: object

    def plan(self):
        owner, spec = self.owner, self.owner.spec
        args = (
            self.indptr,
            self.indices,
            self.last_page_len,
            spec.q_heads,
            spec.kv_heads,
            spec.head_dim,
            owner.page_size,
        )
        kwargs = dict(
            q_data_type=spec.dtype,
            kv_data_type=owner.k.dtype,
            sm_scale=spec.scale,
            window_left=spec.window[0],
            logits_soft_cap=spec.softcap,
        )
        if owner.phase != "decode":
            args = (self.batch.cu_q, *args)
            kwargs["causal"] = spec.causal
        self.wrapper.plan(*args, **kwargs)

    def copy_for_graph(self, new):
        if self.owner.metadata_key != new.owner.metadata_key:
            raise ValueError("FlashInfer Graph semantics changed")
        self.batch.copy_for_graph(new.batch)
        for name in ("indptr", "indices", "last_page_len"):
            old, value = getattr(self, name), getattr(new, name)
            if old.shape != value.shape:
                raise ValueError("FlashInfer Graph buffers must retain their shape")
            if old.data_ptr() != value.data_ptr():
                old.copy_(value, non_blocking=True)
        # Replan before replay, including changes to sequence lengths and split scheduling.
        self.plan()


class BoundFlashInferAttention(BoundAttention):
    @property
    def metadata_key(self):
        return self.name, self.spec, self.view.format

    def __init__(self, offer, state_view, static_params, workspace):
        super().__init__(offer, state_view, static_params, workspace)
        self.scales = (None, None)
        if state_view.format.encoding == "fp8_per_tensor":
            self.page_size = 1
            # FlashInfer accepts scalar calibration. Read it once during binding.
            self.scales = tuple(state_view.fields[2].tolist())
        self.cache = tuple(t.view(-1, self.page_size, self.spec.kv_heads, self.spec.head_dim) for t in (self.k, self.v))

    def create_metadata(self, batch):
        import flashinfer
        from lightllm.common.basemodel.triton_kernel.repack_kv_index import repack_kv_index

        if batch.batch_size > self.limits.max_batch_size or batch.max_kv_len > self.limits.max_seq_len:
            raise ValueError("batch exceeds the planned attention capacity")
        if self.phase == "verify":
            batch = group_verify_batch(batch, self.limits.max_verify_tokens)
        scratch = self._lane_workspace(batch.microbatch_index)
        float_workspace = scratch[:FLOAT_WORKSPACE_BYTES]
        int_workspace = scratch[FLOAT_WORKSPACE_BYTES : FLOAT_WORKSPACE_BYTES + INT_WORKSPACE_BYTES]
        batch = replace(batch, cu_q=batch.cu_q.int(), cu_k=batch.cu_k.int())
        page_counts = (batch.seq_lengths + self.page_size - 1) // self.page_size
        indptr = torch.zeros(batch.batch_size + 1, dtype=torch.int32, device=self.k.device)
        torch.cumsum(page_counts, dim=0, dtype=torch.int32, out=indptr[1:])
        last_page_len = batch.seq_lengths - (page_counts - 1) * self.page_size
        last_page_len = torch.where(page_counts == 0, 0, last_page_len).int()
        width = (self.limits.max_seq_len + self.page_size - 1) // self.page_size
        indices = scratch[FLOAT_WORKSPACE_BYTES + INT_WORKSPACE_BYTES :].view(torch.int32)[: batch.batch_size * width]
        repack_kv_index(
            batch.req_to_tokens,
            batch.req_indices,
            batch.seq_lengths,
            indptr[:-1],
            batch.max_kv_len,
            indices,
            self.page_size,
        )
        if self.phase != "decode":
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                float_workspace,
                "NHD",
                use_cuda_graph=True,
                qo_indptr_buf=batch.cu_q,
                paged_kv_indptr_buf=indptr,
                paged_kv_indices_buf=indices,
                paged_kv_last_page_len_buf=last_page_len,
                backend="fa2",
            )
        else:
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                float_workspace,
                "NHD",
                use_cuda_graph=True,
                use_tensor_cores=True,
                paged_kv_indptr_buffer=indptr,
                paged_kv_indices_buffer=indices,
                paged_kv_last_page_len_buffer=last_page_len,
                backend="fa2",
            )
        wrapper.reset_workspace_buffer(float_workspace, int_workspace)
        metadata = FlashInferBatchMetadata(self, batch, indptr, indices, last_page_len, wrapper)
        metadata.plan()
        return metadata

    def run(self, inputs, metadata, out=None):
        inputs = query_tensor(inputs)
        if inputs.dtype != self.spec.dtype or metadata.owner.metadata_key != self.metadata_key:
            raise ValueError("FlashInfer inputs or metadata differ from the bound semantics")
        out = self.alloc_tensor(inputs.shape, inputs.dtype) if out is None else out
        metadata.wrapper.run(inputs, self.cache, k_scale=self.scales[0], v_scale=self.scales[1], out=out)
        return out


class FlashInferAttentionBackend:
    implementation = "flashinfer"

    def __init__(self, phase):
        if phase not in ("prefill", "decode", "verify"):
            raise ValueError("FlashInfer Dense supports prefill, decode and verify")
        self.phase, self.name = phase, f"flashinfer.{phase}"

    def probe(self, attention_spec, format_spec, device_caps, limits):
        if not isinstance(attention_spec, DenseAttentionSpec):
            return ProbeResult(reason="requires Dense attention semantics")
        if attention_spec.attention_sink:
            return ProbeResult(reason="this FlashInfer implementation does not implement attention sinks")
        if not device_caps.is_nvidia or device_caps.capability is None or device_caps.capability[0] < 8:
            return ProbeResult(reason="FlashInfer requires SM80 or newer CUDA devices")
        if find_spec("flashinfer") is None:
            return ProbeResult(reason="FlashInfer is not installed")
        spec, format = attention_spec, format_spec
        if (
            spec.dtype not in (torch.float16, torch.bfloat16)
            or format.layout != "token_head_dim"
            or format.version != 1
        ):
            return ProbeResult(reason="requires FP16/BF16 queries and token/head/dim storage")
        if format.encoding == "plain":
            if format.dtype != spec.dtype or format.fields != ("k", "v"):
                return ProbeResult(reason="plain KV must match the query dtype")
        elif format.encoding == "fp8_per_tensor":
            if (
                format.dtype != torch.float8_e4m3fn
                or format.scale_dtype != torch.float32
                or format.fields != ("k", "v", "kv_scale", "q_scale")
            ):
                return ProbeResult(reason="requires static E4M3 per-tensor KV with FP32 scales")
        else:
            return ProbeResult(reason="supports plain and static per-tensor FP8 KV")
        if (
            spec.head_dim not in (64, 128, 256)
            or not spec.causal
            or (spec.window != (-1, -1) and not (spec.window[0] >= 0 and spec.window[1] == 0))
        ):
            return ProbeResult(reason="requires causal Dense with supported head dimension and optional left window")
        page_size = 1 if format.encoding == "fp8_per_tensor" else format.page_size
        indices_bytes = limits.max_batch_size * ((limits.max_seq_len + page_size - 1) // page_size) * 4
        return ProbeResult(
            offers=(
                AttentionOffer(
                    self.name,
                    self.phase,
                    format,
                    (FLOAT_WORKSPACE_BYTES + INT_WORKSPACE_BYTES + indices_bytes) * limits.microbatch_count,
                    # The integer plan must remain valid across all layers using this metadata.
                    workspace_group=f"{spec!r}:{format!r}",
                    execution_limits=limits,
                ),
            )
        )

    def bind(self, offer, state_view, static_params, workspace):
        if offer.backend != self.name or offer.phase != self.phase or offer.format != state_view.format:
            raise ValueError("offer does not match the backend and state view")
        k, v = state_view.fields[:2]
        spec = static_params.spec
        if k.ndim != 3 or k.shape != v.shape or k.shape[1:] != (spec.kv_heads, spec.head_dim):
            raise ValueError("KV shape differs from attention semantics")
        if any(t.dtype != state_view.format.dtype or t.device != k.device or not t.is_contiguous() for t in (k, v)):
            raise ValueError("KV fields require the declared dtype and contiguous storage on the same device")
        if state_view.format.encoding == "fp8_per_tensor":
            scales = state_view.fields[2]
            if (
                scales is None
                or scales.shape != (2,)
                or scales.dtype != torch.float32
                or scales.device != k.device
                or not bool((torch.isfinite(scales) & (scales > 0)).all())
            ):
                raise ValueError("FP8 requires two positive finite FP32 calibration scales")
        if (
            workspace is None
            or workspace.dtype != torch.uint8
            or workspace.numel() < offer.workspace_bytes
            or workspace.device != k.device
            or not workspace.is_contiguous()
        ):
            raise ValueError("workspace does not satisfy the selected offer")
        return BoundFlashInferAttention(offer, state_view, static_params, workspace)
