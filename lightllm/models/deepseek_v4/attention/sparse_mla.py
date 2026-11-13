from lightllm.common.attention.api.spec import CacheEncoding, AttentionFamily, AttentionPhase, AttentionFeature
import inspect
from dataclasses import dataclass

import torch

from lightllm.common.attention.api import UnsupportedAttention
from lightllm.common.attention.api.backend import AttentionContext, AttentionBackend


@dataclass(frozen=True)
class DeepseekV4Resources:
    swa: torch.Tensor
    prefill_swa: torch.Tensor
    compressed: torch.Tensor | None
    compress_ratio: int
    head_dim_v: int
    softmax_scale: float
    attn_sink: torch.Tensor


@dataclass(frozen=True)
class DeepseekV4Inputs:
    query: torch.Tensor
    swa_indices: torch.Tensor
    swa_lengths: torch.Tensor
    extra_indices: torch.Tensor | None = None
    extra_lengths: torch.Tensor | None = None


@dataclass
class SwaPrefillWorkspace:
    swa_buffer: torch.Tensor
    HOLD_SWA_SLOT: int
    prefill_token_ranges: torch.Tensor
    swa_prefill_workspace_offsets: torch.Tensor


def prepare_swa_workspace(swa_state, buffer, capacity, batch, slot):
    from lightllm.models.deepseek_v4.triton_kernel.swa_cache import prepare_swa_prefill_workspace

    metadata = SwaPrefillWorkspace(
        buffer,
        buffer.shape[1] - swa_state.config.page_token_size,
        torch.zeros((swa_state.request_capacity, 2), dtype=torch.int32, device=buffer.device),
        torch.full((swa_state.request_capacity,), -1, dtype=torch.int32, device=buffer.device),
    )
    prepare_swa_prefill_workspace(
        swa_state=swa_state,
        prefill_state=metadata,
        b_req_idx=batch.b_req_idx,
        b_ready_cache_len=batch.b_ready_cache_len,
        b_seq_len=batch.b_seq_len,
        workspace_base=slot * capacity,
    )
    return metadata


@dataclass(slots=True)
class DeepseekV4Context(AttentionContext):
    sched_meta_by_ratio: dict | None = None
    capture_sched_meta_by_ratio: dict | None = None


class DeepseekV4SparseMLAFp8Common:
    variant = "deepseek_v4"
    family = AttentionFamily.SPARSE_MLA

    provider = "flashmla_sparse"

    dependency = "flash_mla"

    min_device_capability = (9, 0)

    activation_dtypes = frozenset({torch.float16, torch.bfloat16})

    cache_encodings = frozenset({CacheEncoding.FP8_SPARSE_MLA})

    @classmethod
    def check_environment(cls, caps) -> None:
        super().check_environment(caps)
        import flash_mla

        parameters = inspect.signature(flash_mla.flash_mla_with_kvcache).parameters
        required = {"out", "attn_sink", "indices", "topk_length", "extra_k_cache", "extra_topk_length"}
        missing = required.difference(parameters)
        if missing:
            raise UnsupportedAttention(
                "DSV4 requires the FlashMLA version pinned in docker/Dockerfile; missing: " + ", ".join(sorted(missing))
            )

    def __init__(self, env, spec):
        super().__init__(env=env, spec=spec)
        from lightllm.models.deepseek_v4.config import create_attention_config

        config = create_attention_config(
            model_config=env.network_config,
            model_logical_layer_ids=env.layer_ids,
            tp_rank=env.tp_rank,
            tp_world_size=env.tp_size,
        )
        self.compress_ratios = tuple(dict.fromkeys(config.compress_ratios))
        self.swa_config = config.create_swa_config()
        self.compressed_configs = {
            ratio: config.create_compressed_config(compress_ratio=ratio) for ratio in self.compress_ratios if ratio
        }

    def _plan(self, context):
        import flash_mla as flashmla

        if context.sched_meta_by_ratio is None:
            context.sched_meta_by_ratio = {ratio: flashmla.get_mla_metadata()[0] for ratio in self.compress_ratios}
            context.capture_sched_meta_by_ratio = {
                ratio: flashmla.get_mla_metadata()[0] for ratio in self.compress_ratios
            }

    def _run(self, layer, context, inputs, swa, out):
        import flash_mla as flashmla

        params = layer.resources
        compressed = None
        if params.compressed is not None:
            compressed = self.compressed_configs[params.compress_ratio].get_attention_view(packed_kv=params.compressed)
        scheduler = (
            context.capture_sched_meta_by_ratio
            if torch.cuda.is_current_stream_capturing()
            else context.sched_meta_by_ratio
        )
        q = inputs.query
        if out is None:
            out = self.alloc_tensor((q.shape[0], q.shape[1], params.head_dim_v), dtype=q.dtype, device=q.device)
        padded_out, _ = flashmla.flash_mla_with_kvcache(
            q=q.unsqueeze(1),
            k_cache=self.swa_config.get_attention_view(packed_kv=swa),
            block_table=None,
            cache_seqlens=None,
            head_dim_v=params.head_dim_v,
            tile_scheduler_metadata=scheduler[params.compress_ratio],
            num_splits=None,
            softmax_scale=params.softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=inputs.swa_indices,
            attn_sink=params.attn_sink,
            topk_length=inputs.swa_lengths,
            extra_k_cache=compressed,
            extra_indices_in_kvcache=inputs.extra_indices,
            extra_topk_length=inputs.extra_lengths,
            out=out.unsqueeze(1),
        )
        return padded_out[:, 0, : self.spec.num_q_heads]


class DeepseekV4SparseMLAFp8PrefillBackend(
    DeepseekV4SparseMLAFp8Common, AttentionBackend[DeepseekV4Context, DeepseekV4Inputs, torch.Tensor]
):
    phase = AttentionPhase.PREFILL
    supported_features = frozenset(
        {AttentionFeature.MTP, AttentionFeature.DYNAMIC_VERIFY, AttentionFeature.SINK, AttentionFeature.MTP_WINDOW}
    )

    def allocate_context(self, capacity, *, capture, workspace) -> DeepseekV4Context:
        return DeepseekV4Context(capacity, capture, workspace)

    def plan_host(self, context: DeepseekV4Context):
        self._plan(context)

    def forward(self, layer, context: DeepseekV4Context, inputs: DeepseekV4Inputs, *, out=None) -> torch.Tensor:
        return self._run(layer, context, inputs, layer.resources.prefill_swa, out)


class DeepseekV4SparseMLAFp8DecodeBackend(
    DeepseekV4SparseMLAFp8Common, AttentionBackend[DeepseekV4Context, DeepseekV4Inputs, torch.Tensor]
):
    phase = AttentionPhase.DECODE
    supported_features = frozenset(
        {
            AttentionFeature.MTP,
            AttentionFeature.DYNAMIC_VERIFY,
            AttentionFeature.SINK,
            AttentionFeature.NONCAUSAL,
            AttentionFeature.MTP_WINDOW,
        }
    )

    def allocate_context(self, capacity, *, capture, workspace) -> DeepseekV4Context:
        return DeepseekV4Context(capacity, capture, workspace)

    def plan_host(self, context: DeepseekV4Context):
        self._plan(context)

    def forward(self, layer, context: DeepseekV4Context, inputs: DeepseekV4Inputs, *, out=None) -> torch.Tensor:
        return self._run(layer, context, inputs, layer.resources.swa, out)
