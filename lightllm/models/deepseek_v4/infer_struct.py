import dataclasses

import torch
from lightllm.common.basemodel import InferStateInfo
from lightllm.common.req_manager import DeepseekV4ReqManager
from lightllm.common.kv_cache_mem_manager import DeepseekV4MemoryManager

C4_PREFILL_LOGITS_BUDGET_BYTES = 512 * 1024 * 1024


@dataclasses.dataclass
class PagedIndexerMetadata:
    """Shared paged scorer inputs for the C4 layers of one forward."""

    compress_ratio: int
    row_page_table: torch.Tensor
    logits_page_table: torch.Tensor
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
        self.indexer_metadata: PagedIndexerMetadata | None = None

    def init_some_extra_state(self, model):
        self.indexer_metadata = None  # Rebuild for each eager forward or graph capture.
        self.dsv4_c4_indices = None
        self.dsv4_c4_lengths = None
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
        # Sliding-window indices are layer-independent; build once for this forward.
        from lightllm.models.deepseek_v4.triton_kernel.build_swa_indices import (
            build_image_visibility,
            build_swa_indices,
        )

        self.dsv4_workspace = model.dsv4_workspace
        has_vision = model.has_bidirectional_image_attention
        swa_width = self.req_manager.cache_config.sliding_window
        if has_vision and self.is_prefill:
            swa_width += int(model.config.get("vision_max_n_token", 0))
        if (
            self.is_draft_model
            and model.args.mtp_mode == "dspark"
            and not self.is_prefill
            and self.mtp_draft_input_hiddens is None
        ):
            # DSpark proposals include the complete draft block, with FlashMLA padding.
            swa_width = ((swa_width + int(model.args.mtp_step) + 127) // 128) * 128
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
        from lightllm.models.deepseek_v4.triton_kernel.build_compressed_kv_indices import build_compressed_kv_indices

        long_ratio = self.req_manager.cache_config.long_compress_ratio
        max_kv_seq_len = self.max_kv_seq_len
        # Replay copies C128 indices into the captured state, so their widths must match.
        if not self.is_prefill and model.graph is not None and model.graph.can_run(self.batch_size, max_kv_seq_len):
            max_kv_seq_len = model.graph.graph_max_len_in_batch
        if long_ratio == 128:
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

    def init_c4_indices(self, width):
        # Create layer outputs on first use; replay inputs leave them unset, so
        # copy_for_cuda_graph stages only inputs into the captured infer state.
        if self.dsv4_c4_indices is None:
            self.dsv4_c4_indices = torch.empty(
                (self.position_ids.numel(), width), dtype=torch.int32, device=self.position_ids.device
            )
            self.dsv4_c4_lengths = torch.empty_like(self.position_ids, dtype=torch.int32)

    def get_indexer_metadata(self, layer_index, capacity, *, dense_prefill=False):
        """Build causal page tables and schedulers once per compression layout."""
        config = self.req_manager.cache_config
        ratio = config.compress_rates[layer_index]
        if self.indexer_metadata is not None and self.indexer_metadata.compress_ratio == ratio:
            return self.indexer_metadata
        import deep_gemm
        from .triton_kernel.build_indexer_page_table import build_indexer_page_table
        from .triton_kernel.cache_io import INDEXER_PAGE_SIZE, get_indexer_k_buffer

        valid_len = ((self.position_ids + 1) // ratio).to(torch.int32)
        valid_len.masked_fill_(self.dsv4_sparse_req_idx == self.req_manager.HOLD_REQUEST_ID, 0)
        ctx_lens = valid_len.clamp_min(1).view(-1, 1)
        if ratio == 4:
            torch.clamp(valid_len, min=1, max=self.dsv4_c4_indices.shape[1], out=self.dsv4_c4_lengths)
        page_table = build_indexer_page_table(
            self.dsv4_sparse_req_idx,
            valid_len,
            capacity,
            self.req_manager.req_to_token_indexs,
            self.req_manager.HOLD_REQUEST_ID,
            compress_ratio=ratio,
        )
        logits_page_table = page_table
        entries = config.compressed_kv_token_alignment // ratio
        if entries > INDEXER_PAGE_SIZE:
            packed = get_indexer_k_buffer(self.mem_manager, layer_index)
            slabs = entries // INDEXER_PAGE_SIZE
            logits_page_table = (page_table // slabs) * (packed.stride(0) // config.indexer_head_dim) + (
                page_table % slabs
            ) * (INDEXER_PAGE_SIZE * config.indexer_bytes // config.indexer_head_dim)
        rows_per_chunk = max(1, self.position_ids.numel())
        if self.is_prefill:
            aligned_capacity = ((capacity + 255) // 256) * 256
            rows_per_chunk = max(1, C4_PREFILL_LOGITS_BUDGET_BYTES // (aligned_capacity * 4))
        num_sms = deep_gemm.get_num_sms()
        schedulers = (
            ()
            if dense_prefill
            else tuple(
                deep_gemm.get_paged_mqa_logits_metadata(
                    ctx_lens[start : start + rows_per_chunk], INDEXER_PAGE_SIZE, num_sms
                )
                for start in range(0, self.position_ids.numel(), rows_per_chunk)
            )
        )
        self.indexer_metadata = PagedIndexerMetadata(
            compress_ratio=ratio,
            row_page_table=page_table,
            logits_page_table=logits_page_table,
            valid_lengths=valid_len,
            context_lengths=ctx_lens,
            rows_per_chunk=rows_per_chunk,
            schedulers=schedulers,
        )
        return self.indexer_metadata
