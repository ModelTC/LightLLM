from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import (
    DSV4_SWA_PAGE_SIZE,
)
from lightllm.models.deepseek_v4.infer_struct import DeepseekV4InferStateInfo
from lightllm.models.deepseek_v4.triton_kernel.build_dspark_swa_index import (
    build_dspark_swa_index,
)


class DeepseekV4DSparkInferStateInfo(DeepseekV4InferStateInfo):
    """DeepSeek-V4 metadata with non-causal visibility across one DSpark block."""

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        if self.is_prefill or self.mtp_draft_swa_pages is None:
            # Target-hidden commit passes use target-owned mappings. Proposal
            # blocks, including CUDA Graph HOLD capture, always carry scratch
            # pages and replace these base indices below.
            return

        (
            self.dsv4_swa_indices,
            self.dsv4_swa_lengths,
            self.dsv4_swa_write_slots,
        ) = model.dsv4_workspace.dspark_swa(
            self.microbatch_index,
            self.position_ids.numel(),
        )
        build_dspark_swa_index(
            req_idx=self.dsv4_sparse_req_idx,
            positions=self.position_ids,
            req_to_token_indexs=self.req_manager.req_to_token_indexs,
            full_to_swa_indexs=self.mem_manager.full_to_swa_indexs,
            scratch_pages=self.mtp_draft_swa_pages,
            swa_index=self.dsv4_swa_indices,
            swa_length=self.dsv4_swa_lengths,
            swa_write_slots=self.dsv4_swa_write_slots,
            window=model.config["sliding_window"],
            block_size=model.block_size,
            page_size=DSV4_SWA_PAGE_SIZE,
            hold_req_id=self.req_manager.HOLD_REQUEST_ID,
            hold_full_slot=self.mem_manager.HOLD_TOKEN_MEMINDEX,
            hold_swa_slot=self.mem_manager.swa_pool.HOLD_TOKEN_MEMINDEX,
        )
