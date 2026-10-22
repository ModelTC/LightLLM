from lightllm.models.deepseek_v4.infer_struct import DeepseekV4InferStateInfo
from lightllm.models.deepseek_v4.triton_kernel.build_dspark_swa_index import (
    build_dspark_swa_index,
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
            state_manager=self.state_manager,
            swa_index=self.dsv4_swa_indices,
            swa_length=self.dsv4_swa_lengths,
            swa_write_slots=self.dsv4_swa_write_slots,
            window=model.config["sliding_window"],
            block_size=model.block_size,
            hold_req_id=self.req_manager.HOLD_REQUEST_ID,
            hold_swa_slot=self.state_manager.HOLD_SWA_SLOT,
        )
