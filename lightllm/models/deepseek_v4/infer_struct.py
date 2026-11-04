from functools import partial
import dataclasses
from typing import Literal

import torch
from lightllm.common.basemodel import InferStateInfo
from lightllm.common.execution import batch_metadata
from lightllm.common.execution.batch_metadata import RotaryTables
from lightllm.common.attention_state.bounded_att import FP8E8M0SlidingWindowState, WindowCompressorState
from lightllm.common.attention_state.full_att.compressed_att.compressed_att_state import (
    CompressedAttentionState,
    CompressedAttentionLayerView,
)

C4_PREFILL_LOGITS_BUDGET_BYTES = 512 * 1024 * 1024


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
    SWA and compressor rings are owned by the model's bounded State.
    """

    @property
    def swa_state(self) -> FP8E8M0SlidingWindowState:
        return self.att_state_manager.get_state(
            model_role=self.att_state_role,
            state_type_filter="swa_",
            logical_layer_id=self.first_logical_layer_id,
            state_class=FP8E8M0SlidingWindowState,
        )

    def get_compressor_state(
        self, logical_layer_id: int, target: Literal["kv", "indexer"] = "kv"
    ) -> WindowCompressorState:
        """按当前层和用途选择独立的窗口压缩器。"""
        return self.att_state_manager.get_state(
            model_role=self.att_state_role,
            state_type_filter=f"compressor_{target}_",
            logical_layer_id=logical_layer_id,
            state_class=WindowCompressorState,
        )

    def get_compressed_kv_state(self, logical_layer_id: int) -> CompressedAttentionState:
        return self.att_state_manager.get_state(
            model_role=self.att_state_role,
            state_type_filter="compressed",
            logical_layer_id=logical_layer_id,
            state_class=CompressedAttentionState,
        )

    def get_compressed_kv_view(self, logical_layer_id: int) -> CompressedAttentionLayerView:
        return self.get_compressed_kv_state(logical_layer_id).get_layer_view(logical_layer_id)

    def __init__(self):
        super().__init__()
        self.position_cos_sliding = None
        self.position_sin_sliding = None
        self.position_cos_compress = None
        self.position_sin_compress = None
        # Layer-independent indices are built once per forward in prepare_metadata.
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
        # token -> batch-position map for the compressor; built per prefill forward in prepare_metadata.
        self._dsv4_token_to_batch_idx = None
        # The first C4 layer builds metadata and later layers reuse it within this forward.
        # During CUDA Graph capture its tensor producers become graph nodes. Replay updates
        # those captured buffers from staged inputs; copy_for_cuda_graph skips this container.
        self.c4_indexer_metadata: C4IndexerMetadata | None = None

    @classmethod
    def bind_metadata(cls, model):
        layer_ids = tuple(layer.layer_num_ for layer in model.layers_infer)
        ratios = model.config["compress_ratios"]
        c4_layers = tuple(layer for layer in layer_ids if layer < len(ratios) and ratios[layer] == 4)
        return partial(
            cls.prepare_metadata,
            first_layer=layer_ids[0],
            first_c4_layer=c4_layers[0] if c4_layers else None,
            has_c128_layer=any(layer < len(ratios) and ratios[layer] == 128 for layer in layer_ids),
            sliding=RotaryTables(model._cos_cached_sliding, model._sin_cached_sliding),
            compress=RotaryTables(model._cos_cached_compress, model._sin_cached_compress),
            workspace=model.dsv4_workspace,
            prepare_swa=cls.bind_swa_metadata(model),
            graph_capacity=model.graph_runner.decode_metadata_capacity,
        )

    @classmethod
    def bind_swa_metadata(cls, model):
        swa_state = model.att_state_manager.get_state(
            model_role=model.att_state_role,
            state_type_filter="swa_",
            logical_layer_id=model.layers_infer[0].layer_num_,
            state_class=FP8E8M0SlidingWindowState,
        )
        return partial(
            DeepseekV4InferStateInfo._init_swa_metadata,
            swa_state=swa_state,
            vision_tokens=model.config["vision_max_n_token"] if model.has_vision else None,
        )

    def prepare_metadata(
        self, *, first_layer, first_c4_layer, has_c128_layer, sliding, compress, workspace, prepare_swa, graph_capacity
    ):
        self.first_logical_layer_id = first_layer
        self.first_c4_logical_layer_id = first_c4_layer
        self.c4_indexer_metadata = None  # Rebuild for each eager forward or graph capture.
        self.dsv4_c4_indices = None
        self.dsv4_c4_lengths = None
        self.dsv4_c128_indices = None
        self.dsv4_c128_lengths = None
        batch_metadata.prepare_sequence_metadata(self)
        if self.is_prefill:
            self.prefill_att_state.prepare_swa_workspace(self)
        pos = self.position_ids
        self.position_cos_sliding, self.position_sin_sliding = sliding.gather(pos)
        self.position_cos_compress, self.position_sin_compress = compress.gather(pos)
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
        self.dsv4_workspace = workspace
        prepare_swa(self)

        # 当前模型没有 C128 层时，不分配或构建这组索引；draft 只检查自身拥有的层。
        if not has_c128_layer:
            return

        from lightllm.models.deepseek_v4.triton_kernel.build_compressed_kv_indices import build_compressed_kv_indices

        c128_compress_ratio = 128
        max_kv_seq_len = self.max_kv_seq_len
        # Replay copies C128 indices into the captured state, so their widths must match.
        if not self.is_prefill:
            max_kv_seq_len = graph_capacity(self.batch_size, max_kv_seq_len)
        c128_entries = max(1, max_kv_seq_len // c128_compress_ratio)
        c128_cap = ((c128_entries + 63) // 64) * 64
        self.dsv4_c128_indices = torch.empty((pos.numel(), c128_cap), dtype=torch.int32, device=pos.device)
        self.dsv4_c128_lengths = torch.empty_like(pos, dtype=torch.int32)
        build_compressed_kv_indices(
            self.dsv4_sparse_req_idx,
            self.position_ids,
            self.req_manager.req_to_token_indexs,
            c128_compress_ratio,
            self.dsv4_c128_indices,
            self.dsv4_c128_lengths,
        )

    def _init_swa_metadata(self, *, swa_state, vision_tokens):
        """Build the sliding-window metadata once for this forward."""
        from lightllm.models.deepseek_v4.triton_kernel.build_swa_indices import (
            build_image_visibility,
            build_swa_indices,
        )

        pos = self.position_ids
        swa_width = swa_state.config.window_size
        if vision_tokens is not None and self.is_prefill:
            swa_width += vision_tokens
        self.dsv4_swa_indices = torch.empty((pos.numel(), swa_width), dtype=torch.int32, device=pos.device)
        self.dsv4_swa_lengths = torch.empty_like(pos, dtype=torch.int32)
        if vision_tokens is not None and self.is_prefill:
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
            swa_state=swa_state,
            swa_indices=self.dsv4_swa_indices,
            swa_lengths=self.dsv4_swa_lengths,
            swa_write_slots=self.dsv4_swa_write_slots,
            sliding_window=swa_state.config.window_size,
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
        page_size = self.get_compressed_kv_state(self.first_c4_logical_layer_id).config.block_token_size // 4
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
