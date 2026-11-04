"""FlashInfer MLA plans use explicit, separately reserved buffers for each semantic group."""

from dataclasses import dataclass
from importlib.util import find_spec

import torch

from lightllm.core.compute.attention.dense.flashinfer import FLOAT_WORKSPACE_BYTES, INT_WORKSPACE_BYTES
from lightllm.core.compute.attention.mla.backend import BoundMLAAttention, MLAAttentionBackend
from lightllm.core.contracts.provider import AttentionOffer, ProbeResult
from lightllm.core.compute.attention.mla.spec import MLAAttentionSpec


@dataclass
class FlashInferMLAMetadata:
    owner: object
    batch: object
    indices: torch.Tensor | None
    wrapper: object
    plan_layout: tuple | None = None

    def plan(self):
        owner, batch, spec = self.owner, self.batch, self.owner.spec
        if owner.phase == "prefill":
            self.wrapper.plan(
                batch.cu_q,
                batch.cu_k,
                spec.q_heads,
                spec.q_heads,
                spec.nope_dim + spec.rope_dim,
                head_dim_vo=spec.value_dim,
                causal=True,
                sm_scale=spec.scale,
                q_data_type=spec.dtype,
                kv_data_type=spec.dtype,
            )
        else:
            self.wrapper.plan(
                batch.cu_q,
                batch.cu_k,
                self.indices,
                batch.seq_lengths,
                spec.q_heads,
                spec.latent_dim,
                spec.rope_dim,
                1,
                False,
                spec.scale,
                spec.dtype,
                spec.dtype,
            )
            # FlashInfer captures offsets and launch geometry as scalar kernel arguments.
            # A public replan is safe only while that layout remains identical.
            layout = tuple(self.wrapper._plan_info)
            if self.plan_layout is not None and layout != self.plan_layout:
                raise RuntimeError("FlashInfer MLA Graph plan layout changed; capture a new execution plan")
            self.plan_layout = layout

    def copy_for_graph(self, new):
        if self.owner.metadata_key != new.owner.metadata_key:
            raise ValueError("FlashInfer MLA Graph semantics changed")
        self.batch.copy_for_graph(new.batch)
        if self.indices is not None:
            if new.indices is None or self.indices.shape != new.indices.shape:
                raise ValueError("FlashInfer MLA Graph indices changed shape")
            if self.indices.data_ptr() != new.indices.data_ptr():
                self.indices.copy_(new.indices, non_blocking=True)
        self.plan()


class BoundFlashInferMLA(BoundMLAAttention):
    def __init__(self, offer, view, params, workspace):
        super().__init__(offer, view, params, workspace)
        self.view = view
        self.cache = (
            self.latent[..., : self.spec.latent_dim],
            self.latent[..., self.spec.latent_dim :],
        )

    @property
    def metadata_key(self):
        return self.name, self.spec, self.view.format

    def create_metadata(self, batch):
        import flashinfer
        from lightllm.common.basemodel.triton_kernel.repack_kv_index import repack_kv_index

        batch = super().create_metadata(batch)
        scratch = self._lane_workspace(batch.microbatch_index)
        float_workspace = scratch[:FLOAT_WORKSPACE_BYTES]
        int_workspace = scratch[FLOAT_WORKSPACE_BYTES : FLOAT_WORKSPACE_BYTES + INT_WORKSPACE_BYTES]
        indices = None
        if self.phase == "prefill":
            wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                float_workspace,
                "NHD",
                use_cuda_graph=True,
                qo_indptr_buf=batch.cu_q,
                kv_indptr_buf=batch.cu_k,
                backend="fa2",
            )
            wrapper.reset_workspace_buffer(float_workspace, int_workspace)
        else:
            indices = scratch[FLOAT_WORKSPACE_BYTES + INT_WORKSPACE_BYTES :].view(torch.int32)[
                : batch.batch_size * self.limits.max_seq_len
            ]
            repack_kv_index(
                batch.req_to_tokens,
                batch.req_indices,
                batch.seq_lengths,
                batch.cu_k[:-1],
                batch.max_kv_len,
                indices,
                1,
            )
            wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                float_workspace,
                use_cuda_graph=True,
                qo_indptr=batch.cu_q,
                kv_indptr=batch.cu_k,
                kv_indices=indices,
                kv_len_arr=batch.seq_lengths,
                backend="fa2",
            )
            # The supported MLA wrapper has no public workspace reset method.
            # Bind its integer plan to our reservation before the first plan.
            if wrapper._int_workspace_buffer.numel() != int_workspace.numel():
                raise RuntimeError("FlashInfer MLA integer workspace requirements changed")
            wrapper._int_workspace_buffer = int_workspace
        metadata = FlashInferMLAMetadata(self, batch, indices, wrapper)
        metadata.plan()
        return metadata

    def run(self, inputs, metadata, out=None):
        spec = self.spec
        if metadata.owner.metadata_key != self.metadata_key:
            raise ValueError("FlashInfer MLA metadata differs from the bound semantics")
        if self.phase == "prefill":
            k_nope, k_rope, value = self._expanded_kv(inputs, metadata.batch)
            key = torch.cat((k_nope, k_rope.expand(-1, spec.q_heads, -1)), dim=-1)
            out = self.alloc_tensor((inputs.shape[0], spec.q_heads, spec.value_dim), spec.dtype) if out is None else out
            metadata.wrapper.run(inputs, key, value, out=out)
        else:
            q_nope, q_rope = inputs
            if (
                q_nope.shape != (metadata.batch.batch_size, spec.q_heads, spec.latent_dim)
                or q_rope.shape != (metadata.batch.batch_size, spec.q_heads, spec.rope_dim)
                or q_nope.dtype != spec.dtype
                or q_rope.dtype != spec.dtype
            ):
                raise ValueError("FlashInfer MLA inputs differ from the bound semantics")
            out = self.alloc_tensor(q_nope.shape, spec.dtype) if out is None else out
            metadata.wrapper.run(q_nope, q_rope, *self.cache, out=out)
        return out


class FlashInferMLABackend(MLAAttentionBackend):
    bound_class = BoundFlashInferMLA
    implementation = "flashinfer"

    def __init__(self, phase):
        if phase not in ("prefill", "decode", "verify"):
            raise ValueError("FlashInfer MLA supports prefill, decode and verify")
        self.phase, self.name = phase, f"flashinfer.mla.{phase}"

    def probe(self, attention_spec, format_spec, device_caps, limits):
        if not isinstance(attention_spec, MLAAttentionSpec):
            return ProbeResult(reason="requires MLA semantics")
        if device_caps.device_type != "cuda" or device_caps.capability is None or device_caps.capability[0] < 8:
            return ProbeResult(reason="FlashInfer MLA requires SM80 or newer CUDA devices")
        if find_spec("flashinfer") is None:
            return ProbeResult(reason="FlashInfer is not installed")
        spec, fmt = attention_spec, format_spec
        if (
            fmt.encoding != "plain"
            or fmt.layout != "token_latent"
            or fmt.fields != ("latent",)
            or fmt.version != 1
            or fmt.dtype not in (torch.float16, torch.bfloat16)
            or spec.dtype != fmt.dtype
        ):
            return ProbeResult(reason="requires plain FP16/BF16 latent and rotary storage")
        if spec.rope_dim != 64 or spec.latent_dim not in (128, 256, 512):
            return ProbeResult(reason="unsupported MLA latent or rotary dimensions")
        if self.phase == "prefill" and (spec.nope_dim not in (64, 128, 192) or spec.value_dim not in (64, 128, 256)):
            return ProbeResult(reason="unsupported FlashInfer MLA prefill projections")
        indices_bytes = limits.max_batch_size * limits.max_seq_len * 4 if self.phase != "prefill" else 0
        return ProbeResult(
            offers=(
                AttentionOffer(
                    self.name,
                    self.phase,
                    fmt,
                    (FLOAT_WORKSPACE_BYTES + INT_WORKSPACE_BYTES + indices_bytes) * limits.microbatch_count,
                    workspace_group=f"{spec!r}:{fmt!r}",
                    execution_limits=limits,
                ),
            )
        )
