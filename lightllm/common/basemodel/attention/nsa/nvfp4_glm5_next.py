"""GLM NoPE sparse MLA with an NVFP4 latent and unchanged K-pool indexing."""

import dataclasses
import weakref

import torch

from lightllm.utils.dist_utils import get_current_device_id
from .glm5_next import Glm5NextSparseAttBackend, Glm5NextSparsePrefillState, Glm5NextSparseDecodeState


class NVFP4Glm5NextSparseAttBackend(Glm5NextSparseAttBackend):
    def __init__(self, model):
        super().__init__(model)
        self._prefill_workspaces = self.allocate_prefill_workspaces(model)
        width = ((model.config["index_topk"] + 3 + 127) // 128) * 128
        device = get_current_device_id()
        # Immutable metadata can be shared across layers, microbatches and
        # graphs. Only the per-step causal lengths and selected slots change.
        self.selected_page_table = torch.arange(512 * width, device=device, dtype=torch.int32).view(512, width)
        self.selected_cu_q = torch.arange(513, device=device, dtype=torch.int32)

    @classmethod
    def allocate_prefill_workspaces(cls, model):
        from lightllm.models.glm5_next.triton_kernel.nvfp4_prefill import MAX_STAGED_TOKENS
        from lightllm.models.glm5_next.triton_kernel.nvfp4_selected import _query_tile_size

        regions = 1 + int(model.args.enable_prefill_microbatch_overlap)
        rows = model.batch_max_tokens
        width = ((model.config["index_topk"] + 3 + 127) // 128) * 128
        tokens = min(MAX_STAGED_TOKENS, model.max_req_num * model.max_seq_length + rows)
        if model.args.run_mode == "decode":
            tokens = min(tokens, rows)
        # Selected staging retains padded sparse windows even for tiny contexts.
        tokens = max(tokens, _query_tile_size(rows) * width)
        prefix = f"glm5_nvfp4_prefill_{model.att_state_role}"
        staged = cls.get_gpu_workspace_buffer(prefix + "_kv", regions * tokens * 512, torch.bfloat16).view(
            regions, tokens, 1, 512
        )
        indices = cls.get_gpu_workspace_buffer(prefix + "_topk", regions * rows * width, torch.int32).view(
            regions, rows, width
        )
        lengths = cls.get_gpu_workspace_buffer(prefix + "_seqlens", regions * rows, torch.int32).view(regions, rows)
        cu_q = cls.get_gpu_workspace_buffer(prefix + "_cu_q", regions * (rows + 1), torch.int32).view(regions, rows + 1)
        cu_q.copy_(torch.arange(rows + 1, dtype=torch.int32, device=cu_q.device))
        return [
            dict(
                gathered_kv=staged[index],
                chunk_topk_indices=indices[index],
                chunk_cache_seqlens=lengths[index],
                cu_seqlens_q=cu_q[index],
            )
            for index in range(regions)
        ]

    def get_prefill_workspace(self, infer_state, topk_indices):
        buffers = self._prefill_workspaces[infer_state.microbatch_index]
        rows, width = topk_indices.shape
        assert rows <= buffers["chunk_cache_seqlens"].numel()
        assert width <= buffers["chunk_topk_indices"].shape[1]
        return dict(
            gathered_kv=buffers["gathered_kv"],
            chunk_topk_indices=buffers["chunk_topk_indices"].view(-1)[: rows * width].view(rows, width),
            chunk_cache_seqlens=buffers["chunk_cache_seqlens"][:rows],
            cu_seqlens_q=buffers["cu_seqlens_q"][: rows + 1],
        )

    def create_att_prefill_state(self, infer_state):
        return NVFP4Glm5NextSparsePrefillState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state):
        return NVFP4Glm5NextSparseDecodeState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class NVFP4Glm5NextSparsePrefillState(Glm5NextSparsePrefillState):
    current_row_table: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None

    def init_state(self):
        super().init_state()
        from lightllm.models.glm5_next.triton_kernel.nvfp4_prefill import prefill_slots, prefill_uses_selected

        infer = self.infer_state
        width = self.backend.selected_page_table.shape[1]
        if infer.max_cache_len > 0 and not prefill_uses_selected(
            infer.position_ids.numel(), infer.total_token_num, width
        ):
            self.ragged_mem_index = prefill_slots(
                infer.req_manager.req_to_token_indexs,
                infer.b_req_idx,
                infer.b1_cu_kv_seq_len,
                infer.total_token_num,
            )

    def _nsa_prefill_att(self, q, kv, att_control):
        params = att_control.nsa_prefill_dict
        if self.infer_state.max_cache_len == 0:
            # All visible KV was produced by this chunk: preserve the existing
            # fast BF16 prefill, while the persistent cache is still written FP4.
            from lightllm.models.glm5_next.triton_kernel.nvfp4_prefill import current_chunk_indices

            indices = current_chunk_indices(
                params["topk_indices"], self.q_kv_start_loc, params["prefill_cache_kv"].shape[0]
            )
            control = dataclasses.replace(att_control, nsa_prefill_dict=dict(params, topk_mem_indices=indices))
            return super()._nsa_prefill_att(q, params["prefill_cache_kv"], control)
        row_table = self.current_row_table
        if row_table is None:
            row_table = torch.full((kv.shape[0],), -1, dtype=torch.int32, device=q.device)
            row_table.scatter_(
                0,
                self.infer_state.mem_index.long(),
                torch.arange(self.infer_state.mem_index.numel(), device=q.device, dtype=torch.int32),
            )
            if not torch.cuda.is_current_stream_capturing():
                self.current_row_table = row_table
        from lightllm.models.glm5_next.triton_kernel.nvfp4_prefill import nvfp4_cached_prefill

        return nvfp4_cached_prefill(
            q,
            kv,
            self.ragged_mem_index,
            self.q_kv_start_loc,
            params["topk_indices"],
            params["topk_mem_indices"],
            params["softmax_scale"],
            params["prefill_cache_kv"],
            row_table,
            workspace=self.backend.get_prefill_workspace(self.infer_state, params["topk_indices"]),
        )


@dataclasses.dataclass
class NVFP4Glm5NextSparseDecodeState(Glm5NextSparseDecodeState):
    selected_scheduler_metadata: torch.Tensor = None
    selected_scheduler_offsets: tuple = ()

    def init_state(self):
        super().init_state()
        from lightllm.models.glm5_next.triton_kernel.nvfp4_selected import prepare_selected_metadata

        model = self.backend.model
        self.selected_scheduler_metadata, self.selected_scheduler_offsets = prepare_selected_metadata(
            self.nsa_cache_seqlens,
            model.config["num_attention_heads"] // model.tp_world_size_,
            self.backend.selected_page_table.shape[1],
            self.backend.selected_cu_q,
        )

    def _nsa_decode_att(self, q, kv, att_control):
        params = att_control.nsa_decode_dict
        from lightllm.models.glm5_next.triton_kernel.nvfp4_selected import nvfp4_selected_attention

        return nvfp4_selected_attention(
            q[0],
            kv,
            params["topk_mem_indices"],
            params["softmax_scale"],
            cache_lengths=self.nsa_cache_seqlens,
            page_table=self.backend.selected_page_table,
            cu_q=self.backend.selected_cu_q,
            scheduler_metadata=self.selected_scheduler_metadata,
            scheduler_offsets=self.selected_scheduler_offsets,
        )
