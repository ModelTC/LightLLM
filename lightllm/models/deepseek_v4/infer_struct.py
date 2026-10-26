import dataclasses

import torch
from lightllm.common.basemodel import InferStateInfo
from lightllm.common.req_manager import DeepseekV4ReqManager
from lightllm.common.kv_cache_mem_manager import DeepseekV4MemoryManager
from .workspace import compressed_kv_capacity


@dataclasses.dataclass
class C4IndexerMetadata:
    """Shared paged scorer inputs for the C4 layers of one forward."""

    row_page_table: torch.Tensor
    valid_lengths: torch.Tensor
    context_lengths: torch.Tensor
    scheduler: torch.Tensor | None
    topk_lengths: torch.Tensor
    rows_per_chunk: int | None
    chunk_schedulers: tuple[torch.Tensor, ...] | None


class DeepseekV4InferStateInfo(InferStateInfo):
    """Per-forward inputs, derived indices and attention metadata.

    Views may borrow workspace storage. Request-owned SWA and compressor rings
    remain in the request manager and are never copied into this forward state.
    """

    req_manager: DeepseekV4ReqManager
    mem_manager: DeepseekV4MemoryManager

    def __init__(self):
        super().__init__()
        self.position_cos_sliding = None
        self.position_sin_sliding = None
        self.position_cos_compress = None
        self.position_sin_compress = None
        # layer-independent sparse-index metadata, built once per forward in init_some_extra_state
        # (None until then so copy_for_cuda_graph's tensor-attr loop skips them).
        self.dsv4_sparse_req_idx = None
        self.dsv4_swa_indices = None
        self.dsv4_swa_lengths = None
        self.dsv4_swa_write_slots = None
        self.dsv4_swa_runtime_write_slots = None
        self.dsv4_image_left = None
        self.dsv4_image_right = None
        self.dsv4_c128_indices = None
        self.dsv4_c128_lengths = None
        self.dsv4_workspace = None
        # token -> batch-position map for the compressor; built per prefill forward in init_some_extra_state.
        self._dsv4_token_to_batch_idx = None
        # The first C4 layer builds metadata and later layers reuse it within this forward.
        # During CUDA Graph capture its tensor producers become graph nodes. Replay updates
        # those captured buffers from staged inputs; copy_for_cuda_graph skips this container.
        self.c4_indexer_metadata: C4IndexerMetadata | None = None

    def _dsv4_index_max_kv_seq_len(self, model):
        if (
            not self.is_prefill
            and model.graph is not None
            and model.graph.can_run(self.batch_size, self.max_kv_seq_len)
        ):
            return model.graph.graph_max_len_in_batch
        return self.max_kv_seq_len

    def init_some_extra_state(self, model):
        self.c4_indexer_metadata = None  # Rebuild for each eager forward or graph capture.
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
        # Sliding-window indices are layer-independent, so build them once into the model workspace.
        from lightllm.models.deepseek_v4.triton_kernel.build_swa_indices import (
            build_image_visibility,
            build_swa_indices,
        )

        workspace = model.dsv4_workspace
        self.dsv4_workspace = workspace
        has_vision = model.has_vision
        swa_width = workspace.vision_swa_width if has_vision and self.is_prefill else workspace.sliding_window
        self.dsv4_swa_indices, self.dsv4_swa_lengths = workspace.swa(
            self.microbatch_index, pos.numel(), width=swa_width
        )
        if has_vision and self.is_prefill:
            self.dsv4_image_left = torch.zeros_like(pos, dtype=torch.int32)
            self.dsv4_image_right = torch.zeros_like(pos, dtype=torch.int32)
            image_spans = []
            for batch_id, params in enumerate(self.multimodal_params):
                for image in params["images"]:
                    # start_idx points at IMAGE_START; token_num also includes three canonical alignment slots.
                    image_spans.append((batch_id, image["start_idx"], image["token_num"] - 3))
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
        self.dsv4_swa_indices, self.dsv4_swa_lengths = build_swa_indices(
            token_req_idx=self.dsv4_sparse_req_idx,
            position_ids=self.position_ids,
            req_manager=self.req_manager,
            swa_indices=self.dsv4_swa_indices,
            swa_lengths=self.dsv4_swa_lengths,
            swa_write_slots=self.dsv4_swa_write_slots,
            sliding_window=workspace.sliding_window,
            image_left=self.dsv4_image_left,
            image_right=self.dsv4_image_right,
            prefill_state=self.prefill_att_state if self.is_prefill else None,
            swa_runtime_write_slots=self.dsv4_swa_runtime_write_slots,
            token_to_batch_idx=self._dsv4_token_to_batch_idx,
            b_seq_len=self.b_seq_len,
        )
        from lightllm.models.deepseek_v4.triton_kernel.build_compressed_kv_indices import build_compressed_kv_indices

        long_ratio = self.req_manager.cache_config.long_compress_ratio
        cap = compressed_kv_capacity(self._dsv4_index_max_kv_seq_len(model), long_ratio)
        self.dsv4_c128_indices, self.dsv4_c128_lengths = workspace.c128(self.microbatch_index, pos.numel(), cap)
        build_compressed_kv_indices(
            self.dsv4_sparse_req_idx,
            self.position_ids,
            self.req_manager.req_to_token_indexs,
            long_ratio,
            self.dsv4_c128_indices,
            self.dsv4_c128_lengths,
        )
