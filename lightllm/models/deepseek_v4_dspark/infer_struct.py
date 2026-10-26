from lightllm.models.deepseek_v4.infer_struct import DeepseekV4InferStateInfo
from lightllm.models.deepseek_v4.triton_kernel.build_dspark_swa_indices import (
    build_dspark_swa_indices,
)


class DeepseekV4DSparkInferStateInfo(DeepseekV4InferStateInfo):
    """DeepSeek-V4 metadata with non-causal visibility across one DSpark block."""

    # Produced by the first context layer after DP rebalance, consumed per stage.
    context_kv: tuple

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        if self.is_prefill or self.mtp_draft_input_hiddens is not None:
            # Context commits use causal target positions. Proposals attend to
            # history plus the complete block in the ring's future slots.
            return

        workspace = self.dsv4_workspace
        self.dsv4_swa_indices, self.dsv4_swa_lengths = workspace.swa(
            self.microbatch_index,
            self.position_ids.numel(),
            width=workspace.swa_capacity,
        )
        # The parent forward already owns the write-slot buffer. Rebuild it for
        # draft visibility instead of replacing it with a second workspace copy.
        build_dspark_swa_indices(
            token_req_idx=self.dsv4_sparse_req_idx,
            position_ids=self.position_ids,
            req_manager=self.req_manager,
            swa_indices=self.dsv4_swa_indices,
            swa_lengths=self.dsv4_swa_lengths,
            swa_write_slots=self.dsv4_swa_write_slots,
            sliding_window=model.config["sliding_window"],
            block_size=model.block_size,
            hold_req_id=self.req_manager.HOLD_REQUEST_ID,
            hold_swa_slot=self.req_manager.HOLD_SWA_SLOT,
        )
