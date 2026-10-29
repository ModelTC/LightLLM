"""Projection and quantization scratch for DeepSeek-V4 model computation.

Request managers own persistent rings; infer states build each forward's
indices and attention metadata. This module only holds computation scratch.
"""

import math
import os

import deep_gemm

import torch


C4_PREFILL_LOGITS_BUDGET_BYTES = 256 * 1024 * 1024


class DeepseekV4C4PrefillWorkspace:
    """Persistent scratch used exclusively by the C4 prefill auxiliary stream."""

    def __init__(self, token_capacity: int, index_n_heads: int, index_head_dim: int, max_seq_length: int):
        self.token_capacity = int(token_capacity)
        self.index_n_heads = int(index_n_heads)
        self.index_head_dim = int(index_head_dim)

        max_indexer_k_phase = self.token_capacity * self.index_head_dim * (4 * 4 + 2 + 2)
        max_indexer_q_phase = self.token_capacity * self.index_n_heads * (self.index_head_dim * 2 + 2)

        # K, Q and scoring execute sequentially on the same auxiliary stream.
        self.logits_out_supported = "out:" in (deep_gemm.fp8_paged_mqa_logits.__doc__ or "")
        logits_width = ((max_seq_length // 4 + 255) // 256) * 256
        logits_bytes = min(C4_PREFILL_LOGITS_BUDGET_BYTES, self.token_capacity * logits_width * 4)
        scratch_bytes = max(max_indexer_k_phase, max_indexer_q_phase)
        if self.logits_out_supported:
            scratch_bytes = max(scratch_bytes, logits_bytes)
        self.scratch = torch.empty((scratch_bytes,), dtype=torch.uint8, device="cuda")
        self.idx_q_fp8 = torch.empty(
            (self.token_capacity, self.index_n_heads, self.index_head_dim),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        self.weights = torch.empty((self.token_capacity, self.index_n_heads, 1), dtype=torch.float32, device="cuda")

    def _scratch_view(self, shape, dtype: torch.dtype, byte_offset: int = 0) -> torch.Tensor:
        nbytes = math.prod(shape) * dtype.itemsize
        return self.scratch[byte_offset : byte_offset + nbytes].view(dtype).view(shape)

    def logits(self, row_num: int, c4_cap: int):
        width = ((c4_cap + 255) // 256) * 256
        return self._scratch_view((row_num, width), torch.float32)[:, :c4_cap]

    def indexer_k_buffers(self, token_num: int):
        kv_score_shape = (token_num, 4 * self.index_head_dim)
        indexer_k_shape = (token_num, self.index_head_dim)
        kv_score = self._scratch_view(kv_score_shape, torch.float32)
        offset = kv_score.numel() * kv_score.element_size()
        indexer_k = self._scratch_view(indexer_k_shape, torch.bfloat16, offset)
        offset += indexer_k.numel() * indexer_k.element_size()
        hadamard = self._scratch_view(indexer_k_shape, torch.bfloat16, offset)
        return kv_score, indexer_k, hadamard

    def indexer_q_inputs(self, token_num: int):
        idx_q_shape = (token_num, self.index_n_heads * self.index_head_dim)
        raw_weights_shape = (token_num, self.index_n_heads)
        idx_q = self._scratch_view(idx_q_shape, torch.bfloat16)
        offset = idx_q.numel() * idx_q.element_size()
        raw_weights = self._scratch_view(raw_weights_shape, torch.bfloat16, offset)
        return idx_q, raw_weights

    def indexer_q_outputs(self, token_num: int):
        return self.idx_q_fp8[:token_num], self.weights[:token_num]


class DeepseekV4Workspace:
    """Reusable model computation buffers; forward metadata lives in infer states."""

    def __init__(self, model):
        # Eager prefill and CUDA graph buckets may pad tokens to a full TP group.
        tp = model.tp_world_size_
        self.token_capacity = ((int(model.batch_max_tokens) + tp - 1) // tp) * tp
        self.flashmla_prefill_q = None
        self.c4_prefill_aux = None
        self.flashmla_prefill_o_accum = None
        self.flashmla_prefill_lse_accum = None

    def init_persistent_scratch(self, model):
        # Allocate after the KV pool, using the remaining runtime headroom.
        overlap = model.args.enable_decode_microbatch_overlap or model.args.enable_prefill_microbatch_overlap
        regions = 1 + int(overlap)
        self.c128_compress_scratch = torch.empty(
            (regions, self.token_capacity, model.config["head_dim"]), dtype=torch.float32, device="cuda"
        )
        if model.run_mode == "decode":
            return
        if (
            os.getenv("LIGHTLLM_DSV4_PREFILL_OVERLAP", "1") == "1"
            and not model.args.enable_prefill_microbatch_overlap
            and 4 in model.config["compress_ratios"]
            and model.run_mode == "prefill"
        ):
            self.init_c4_prefill_aux(model)
        if hasattr(torch.ops._flashmla_C, "sparse_decode_fwd_with_workspace"):
            heads = ((model.config["num_attention_heads"] // model.tp_world_size_ + 63) // 64) * 64
            splits = self.token_capacity + torch.cuda.get_device_properties().multi_processor_count
            shape = (regions, splits, 1, heads)
            o_shape = (*shape, model.config["head_dim"])
            o_bytes = math.prod(o_shape) * torch.float32.itemsize
            lse_bytes = math.prod(shape) * torch.float32.itemsize
            if self.c4_prefill_aux is not None:
                # The auxiliary stream joins before attention; logits and split-KV never overlap.
                arena = torch.empty(
                    max(self.c4_prefill_aux.scratch.numel(), o_bytes + lse_bytes), dtype=torch.uint8, device="cuda"
                )
                self.c4_prefill_aux.scratch = arena
                self.flashmla_prefill_o_accum = arena[:o_bytes].view(torch.float32).view(o_shape)
                self.flashmla_prefill_lse_accum = arena[o_bytes : o_bytes + lse_bytes].view(torch.float32).view(shape)
            else:
                self.flashmla_prefill_o_accum = torch.empty(o_shape, dtype=torch.float32, device="cuda")
                self.flashmla_prefill_lse_accum = torch.empty(shape, dtype=torch.float32, device="cuda")

    def init_c4_prefill_aux(self, model):
        self.c4_prefill_aux = DeepseekV4C4PrefillWorkspace(
            token_capacity=self.token_capacity,
            index_n_heads=model.config["index_n_heads"],
            index_head_dim=model.config["index_head_dim"],
            max_seq_length=model.max_seq_length,
        )

    def init_flashmla_prefill_q(self, tp_q_head_num: int, padded_q_head_num: int, head_dim: int, dtype: torch.dtype):
        self.flashmla_prefill_q = torch.empty(
            (self.token_capacity, padded_q_head_num, head_dim), dtype=dtype, device="cuda"
        )
        self.flashmla_prefill_q[:, tp_q_head_num:, :].zero_()

    def get_flashmla_prefill_q(self, token_num: int):
        if token_num > self.token_capacity:
            raise ValueError(
                f"DeepSeek-V4 Q workspace capacity {self.token_capacity} is smaller than {token_num} tokens"
            )
        return self.flashmla_prefill_q[:token_num]
