import dataclasses

import torch
from lightllm.common.basemodel import InferStateInfo
from lightllm.common.req_manager import DeepseekV4ReqManager
from lightllm.common.kv_cache_mem_manager import DeepseekV4MemoryManager

from .workspace import C4_PREFILL_LOGITS_BUDGET_BYTES


@dataclasses.dataclass
class C4IndexerMetadata:
    """Shared paged scorer inputs for the C4 layers of one forward."""

    row_page_table: torch.Tensor
    valid_lengths: torch.Tensor
    context_lengths: torch.Tensor
    rows_per_chunk: int
    schedulers: tuple[torch.Tensor, ...]


class DeepseekV4InferStateInfo(InferStateInfo):
    """Per-forward inputs, derived indices and attention metadata.

    Indices and metadata are built for this batch and reused by its layers.
    Request-owned SWA and compressor rings remain in the request manager.
    """

    req_manager: DeepseekV4ReqManager
    mem_manager: DeepseekV4MemoryManager

    def __init__(self):
        super().__init__()
        self.position_cos_sliding = None
        self.position_sin_sliding = None
        self.position_cos_compress = None
        self.position_sin_compress = None
        # Layer-independent indices are built once per forward in init_some_extra_state.
        self.dsv4_sparse_req_idx = None
        self.dsv4_swa_indices = None
        self.dsv4_swa_lengths = None
        self.dsv4_swa_write_slots = None
        self.dsv4_swa_runtime_write_slots = None
        self.dsv4_image_left = None
        self.dsv4_image_right = None
        self.dsv4_c128_indices = None
        self.dsv4_c128_lengths = None
        self.dsv4_c4_indices = None
        self.dsv4_c4_lengths = None
        self.dsv4_workspace = None
        # token -> batch-position map for the compressor; built per prefill forward in init_some_extra_state.
        self._dsv4_token_to_batch_idx = None
        # The first C4 layer builds metadata and later layers reuse it within this forward.
        # During CUDA Graph capture its tensor producers become graph nodes. Replay updates
        # those captured buffers from staged inputs; copy_for_cuda_graph skips this container.
        self.c4_indexer_metadata: C4IndexerMetadata | None = None

    def init_some_extra_state(self, model):
        self.c4_indexer_metadata = None  # Rebuild for each eager forward or graph capture.
        self.dsv4_c4_indices = None
        self.dsv4_c4_lengths = None
        self.dsv4_workspace = model.dsv4_workspace
        super().init_some_extra_state(model)  # sets position_ids, b_q_seq_len, b_q_start_loc (prefill)
        if self.is_prefill:
            self.prefill_att_state.prepare_swa_workspace(self)
        pos = self.position_ids
        self.position_cos_sliding = torch.index_select(model._cos_cached_sliding, 0, pos)  # [T, rope_dim//2]
        self.position_sin_sliding = torch.index_select(model._sin_cached_sliding, 0, pos)
        self.position_cos_compress = torch.index_select(model._cos_cached_compress, 0, pos)
        self.position_sin_compress = torch.index_select(model._sin_cached_compress, 0, pos)
        # Per-token request id (decode: one token per req; prefill: ragged -> repeat by q-len).
        # Layer-independent; the swa kernel + build_metadata's c4/c128 readers all reuse it.
        if self.is_prefill:
            self.dsv4_sparse_req_idx = torch.repeat_interleave(
                self.b_req_idx, self.b_q_seq_len, output_size=pos.numel()
            )
            self._dsv4_token_to_batch_idx = torch.repeat_interleave(
                torch.arange(self.b_req_idx.shape[0], dtype=torch.int32, device=self.b_req_idx.device),
                self.b_q_seq_len,
                output_size=pos.numel(),
            )
        else:
            self.dsv4_sparse_req_idx = self.b_req_idx
            self._dsv4_token_to_batch_idx = None
        self._init_swa_metadata(model)

        from lightllm.models.deepseek_v4.triton_kernel.build_compressed_kv_indices import build_compressed_kv_indices

        long_ratio = self.req_manager.cache_config.long_compress_ratio
        max_kv_seq_len = self.max_kv_seq_len
        # Replay copies C128 indices into the captured state, so their widths must match.
        if not self.is_prefill and model.graph is not None and model.graph.can_run(self.batch_size, max_kv_seq_len):
            max_kv_seq_len = model.graph.graph_max_len_in_batch
        c128_entries = max(1, max_kv_seq_len // long_ratio)
        c128_cap = ((c128_entries + 63) // 64) * 64
        self.dsv4_c128_indices = torch.empty((pos.numel(), c128_cap), dtype=torch.int32, device=pos.device)
        self.dsv4_c128_lengths = torch.empty_like(pos, dtype=torch.int32)
        build_compressed_kv_indices(
            self.dsv4_sparse_req_idx,
            self.position_ids,
            self.req_manager.req_to_token_indexs,
            long_ratio,
            self.dsv4_c128_indices,
            self.dsv4_c128_lengths,
        )

    def _init_swa_metadata(self, model):
        """Build the sliding-window metadata once for this forward."""
        from lightllm.models.deepseek_v4.triton_kernel.build_swa_indices import (
            build_image_visibility,
            build_swa_indices,
        )

        pos = self.position_ids
        has_vision = model.has_vision
        swa_width = self.req_manager.cache_config.sliding_window
        if has_vision and self.is_prefill:
            swa_width += model.config["vision_max_n_token"]
        self.dsv4_swa_indices = torch.empty((pos.numel(), swa_width), dtype=torch.int32, device=pos.device)
        self.dsv4_swa_lengths = torch.empty_like(pos, dtype=torch.int32)
        if has_vision and self.is_prefill:
            self.dsv4_image_left = torch.zeros_like(pos, dtype=torch.int32)
            self.dsv4_image_right = torch.zeros_like(pos, dtype=torch.int32)
            image_spans = []
            for batch_id, params in enumerate(self.multimodal_params):
                for image in params["images"]:
                    # start_idx includes alignment padding; bidirectional visibility starts at IMAGE_START.
                    image_start = image["start_idx"] + 3 - image["start_idx"] % 4
                    image_spans.append((batch_id, image_start, image["token_num"] - 3))
            if image_spans:
                build_image_visibility(
                    image_spans=torch.tensor(image_spans, dtype=torch.int32).cuda(non_blocking=True),
                    b_q_start_loc=self.b_q_start_loc,
                    b_ready_cache_len=self.b_ready_cache_len,
                    b_q_seq_len=self.b_q_seq_len,
                    image_left=self.dsv4_image_left,
                    image_right=self.dsv4_image_right,
                )
        self.dsv4_swa_write_slots = torch.empty_like(pos, dtype=torch.int32)
        self.dsv4_swa_runtime_write_slots = torch.empty_like(pos, dtype=torch.int32) if self.is_prefill else None
        build_swa_indices(
            token_req_idx=self.dsv4_sparse_req_idx,
            position_ids=self.position_ids,
            req_manager=self.req_manager,
            swa_indices=self.dsv4_swa_indices,
            swa_lengths=self.dsv4_swa_lengths,
            swa_write_slots=self.dsv4_swa_write_slots,
            sliding_window=self.req_manager.cache_config.sliding_window,
            image_left=self.dsv4_image_left,
            image_right=self.dsv4_image_right,
            prefill_state=self.prefill_att_state if self.is_prefill else None,
            swa_runtime_write_slots=self.dsv4_swa_runtime_write_slots,
            token_to_batch_idx=self._dsv4_token_to_batch_idx,
            b_seq_len=self.b_seq_len,
        )

    def init_c4_indices(self, width):
        # Create layer outputs on first use; replay inputs leave them unset, so
        # copy_for_cuda_graph stages only inputs into the captured infer state.
        if self.dsv4_c4_indices is None:
            self.dsv4_c4_indices = torch.empty(
                (self.position_ids.numel(), width), dtype=torch.int32, device=self.position_ids.device
            )
            self.dsv4_c4_lengths = torch.empty_like(self.position_ids, dtype=torch.int32)

    def get_c4_indexer_metadata(self, c4_cap):
        """Build layer-independent scorer inputs once, on the first C4 layer."""
        if self.c4_indexer_metadata is not None:
            return self.c4_indexer_metadata
        import deep_gemm
        from .triton_kernel.build_c4_indexer_page_table import build_c4_indexer_page_table

        valid_len = ((self.position_ids + 1) // 4).to(torch.int32)
        ctx_lens = valid_len.clamp_min(1).view(-1, 1)
        torch.clamp(valid_len, min=1, max=self.dsv4_c4_indices.shape[1], out=self.dsv4_c4_lengths)
        page_table = build_c4_indexer_page_table(
            self.dsv4_sparse_req_idx,
            valid_len,
            c4_cap,
            self.req_manager.req_to_token_indexs,
            self.req_manager.HOLD_REQUEST_ID,
        )
        page_size = self.req_manager.cache_config.compressed_kv_token_alignment // 4
        rows_per_chunk = max(1, self.position_ids.numel())
        if self.is_prefill:
            # DeepGEMM rounds the logits width to a multiple of 256.
            aligned_c4_cap = ((c4_cap + 255) // 256) * 256
            rows_per_chunk = max(1, C4_PREFILL_LOGITS_BUDGET_BYTES // (aligned_c4_cap * 4))
        num_sms = deep_gemm.get_num_sms()
        schedulers = tuple(
            deep_gemm.get_paged_mqa_logits_metadata(ctx_lens[start : start + rows_per_chunk], page_size, num_sms)
            for start in range(0, self.position_ids.numel(), rows_per_chunk)
        )
        self.c4_indexer_metadata = C4IndexerMetadata(
            row_page_table=page_table,
            valid_lengths=valid_len,
            context_lengths=ctx_lens,
            rows_per_chunk=rows_per_chunk,
            schedulers=schedulers,
        )
        return self.c4_indexer_metadata
