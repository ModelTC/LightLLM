import dataclasses

import torch

from lightllm.models.deepseek_v4.infer_struct import DeepseekV4InferStateInfo
from lightllm.models.deepseek_v4.triton_kernel.build_c4_indexer_page_table import build_c4_indexer_page_table
from lightllm.models.deepseek_v4.triton_kernel.build_swa_indices import build_swa_indices

INDEXER_PREFILL_LOGITS_BUDGET_BYTES = 512 * 1024 * 1024


@dataclasses.dataclass
class PagedIndexerMetadata:
    compress_ratio: int
    row_page_table: torch.Tensor
    valid_lengths: torch.Tensor
    context_lengths: torch.Tensor
    rows_per_chunk: int
    schedulers: tuple[torch.Tensor, ...]
    page_rows: torch.Tensor | None


class DeepseekV41InferStateInfo(DeepseekV4InferStateInfo):
    def __init__(self):
        super().__init__()
        self.indexer_metadata: PagedIndexerMetadata | None = None

    def init_some_extra_state(self, model):
        self.indexer_metadata = None
        super().init_some_extra_state(model)
        self.v41_sparse_selection = None
        self.candidate_block_indices = None
        self.engram_embedding_row_ids = model.engram_hasher.compute_embedding_row_ids(self)

    def _init_swa_metadata(self, model):
        """Build causal sliding-window metadata for both text and image tokens."""
        positions = self.position_ids
        swa_state = self.swa_state
        window_size = swa_state.config.window_size
        self.dsv4_swa_indices = torch.empty(
            (positions.numel(), window_size), dtype=torch.int32, device=positions.device
        )
        self.dsv4_swa_lengths = torch.empty_like(positions, dtype=torch.int32)
        self.dsv4_swa_write_slots = torch.empty_like(positions, dtype=torch.int32)
        self.dsv4_swa_runtime_write_slots = torch.empty_like(positions, dtype=torch.int32) if self.is_prefill else None
        build_swa_indices(
            token_req_idx=self.dsv4_sparse_req_idx,
            position_ids=positions,
            swa_state=swa_state,
            swa_indices=self.dsv4_swa_indices,
            swa_lengths=self.dsv4_swa_lengths,
            swa_write_slots=self.dsv4_swa_write_slots,
            sliding_window=window_size,
            prefill_state=self.prefill_att_state if self.is_prefill else None,
            swa_runtime_write_slots=self.dsv4_swa_runtime_write_slots,
            token_to_batch_idx=self._dsv4_token_to_batch_idx,
            b_seq_len=self.b_seq_len,
        )

    def get_indexer_metadata(self, logical_layer_id, capacity, *, request_pages=False):
        """Reuse scorer inputs for layers using the same compression ratio and page layout."""
        config = self.get_compressed_kv_state(logical_layer_id).config
        ratio = config.compress_ratio
        if (
            self.indexer_metadata is not None
            and self.indexer_metadata.compress_ratio == ratio
            and (self.indexer_metadata.page_rows is not None) == request_pages
        ):
            return self.indexer_metadata
        import deep_gemm

        hold_req_id = self.req_manager.HOLD_REQUEST_ID
        valid_lengths = ((self.position_ids + 1) // ratio).to(torch.int32)
        valid_lengths.masked_fill_(self.dsv4_sparse_req_idx == hold_req_id, 0)
        context_lengths = valid_lengths.clamp_min(1).view(-1, 1)
        page_table = build_c4_indexer_page_table(
            self.b_req_idx if request_pages else self.dsv4_sparse_req_idx,
            self.b_seq_len // ratio if request_pages else valid_lengths,
            capacity,
            self.req_manager.req_to_token_indexs,
            hold_req_id,
            compress_ratio=ratio,
        )
        page_size = config.block_token_size // ratio
        rows_per_chunk = max(1, self.position_ids.numel())
        if self.is_prefill:
            # DeepGEMM rounds the logits width to a multiple of 256.
            aligned_capacity = ((capacity + 255) // 256) * 256
            rows_per_chunk = max(1, INDEXER_PREFILL_LOGITS_BUDGET_BYTES // (aligned_capacity * 4))
        schedulers = ()
        if not request_pages:
            num_sms = deep_gemm.get_num_sms()
            schedulers = tuple(
                deep_gemm.get_paged_mqa_logits_metadata(
                    context_lengths[start : start + rows_per_chunk], page_size, num_sms
                )
                for start in range(0, self.position_ids.numel(), rows_per_chunk)
            )
        self.indexer_metadata = PagedIndexerMetadata(
            compress_ratio=ratio,
            row_page_table=page_table,
            valid_lengths=valid_lengths,
            context_lengths=context_lengths,
            rows_per_chunk=rows_per_chunk,
            schedulers=schedulers,
            page_rows=self._dsv4_token_to_batch_idx if request_pages else None,
        )
        return self.indexer_metadata
