"""GLM-5 NoPE sparse MLA over the packed FP8 (group64, E8M0) cache: FA3 chunked prefill, FlashMLA decode."""

import torch
import triton

from lightllm.common.attention.backend import AttentionStepBuilder
from lightllm.common.attention.backends.sparse_mla.fa3 import FA3SparseMLAPrefillBackend
from lightllm.common.attention.backends.sparse_mla.flashmla import FlashMLASparseDecodeBackend
from .glm5_next import Glm5NextSparseAttBackend


# Cap each microbatch's BF16 KV workspace at 2 GiB (2097152 × 512 × 2 bytes).
KV_GATHER_CHUNK_SIZE = 2097152


class Fp8Glm5NextSparseAttBackend(Glm5NextSparseAttBackend):
    kv_cache_format = "fp8_group64_e8m0"
    prefill_backend = FA3SparseMLAPrefillBackend
    decode_backend = FlashMLASparseDecodeBackend

    def __init__(self, model):
        self._prefill_workspaces = self.allocate_prefill_workspaces(model)
        super().__init__(model=model)

    def _create_prefill_step_builder(self, model) -> AttentionStepBuilder:
        # Prefill scratch is caller-owned and reserved before the KV cache is sized.
        return FA3SparseMLAPrefillBackend.get_step_builder_cls()(
            workspace=lambda microbatch_index: self._prefill_workspaces[microbatch_index]
        )

    def _create_decode_step_builder(self, model) -> AttentionStepBuilder:
        config = model.config
        # 2048 pooled entries plus at most three kpool remainder entries fit the 2112 rounded direct window.
        bound = 2051 if config["index_topk"] == 2048 and config["index_kpool"] == 4 else None
        return FlashMLASparseDecodeBackend.get_step_builder_cls()(direct_decode_bound=bound)

    @classmethod
    def allocate_prefill_workspaces(cls, model):
        regions = 1 + int(model.args.enable_prefill_microbatch_overlap)
        q_token_capacity = model.batch_max_tokens
        topk_size = triton.cdiv(model.config["index_topk"] + 3, 128) * 128
        kv_token_capacity = min(KV_GATHER_CHUNK_SIZE, model.max_req_num * model.max_seq_length + q_token_capacity)
        if model.args.run_mode == "decode":
            kv_token_capacity = min(kv_token_capacity, q_token_capacity)
        prefix = f"glm5_fp8_prefill_{model.att_state_role}"
        gathered_kv = cls.get_gpu_workspace_buffer(
            prefix + "_kv", regions * kv_token_capacity * 512, torch.bfloat16
        ).view(regions, kv_token_capacity, 1, 512)
        chunk_topk_indices = cls.get_gpu_workspace_buffer(
            prefix + "_topk", regions * q_token_capacity * topk_size, torch.int32
        ).view(regions, q_token_capacity, topk_size)
        chunk_cache_seqlens = cls.get_gpu_workspace_buffer(
            prefix + "_seqlens", regions * q_token_capacity, torch.int32
        ).view(regions, q_token_capacity)
        cu_seqlens_q = cls.get_gpu_workspace_buffer(
            prefix + "_cu_q", regions * (q_token_capacity + 1), torch.int32
        ).view(regions, q_token_capacity + 1)
        cu_seqlens_q.copy_(torch.arange(q_token_capacity + 1, dtype=torch.int32, device=cu_seqlens_q.device))
        return [
            {
                "gathered_kv": gathered_kv[index],
                "chunk_topk_indices": chunk_topk_indices[index],
                "chunk_cache_seqlens": chunk_cache_seqlens[index],
                "cu_seqlens_q": cu_seqlens_q[index],
            }
            for index in range(regions)
        ]
