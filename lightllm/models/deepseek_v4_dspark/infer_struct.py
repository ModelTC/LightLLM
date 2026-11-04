from functools import partial
import torch

from lightllm.models.deepseek_v4.infer_struct import DeepseekV4InferStateInfo
from lightllm.models.deepseek_v4.triton_kernel.build_dspark_swa_indices import (
    build_dspark_swa_indices,
)


class DeepseekV4DSparkInferStateInfo(DeepseekV4InferStateInfo):
    """DeepSeek-V4 metadata with non-causal visibility across one DSpark block."""

    # Produced by the first context layer after DP rebalance, consumed per stage.
    context_kv: tuple

    @classmethod
    def bind_swa_metadata(cls, model):
        causal = super().bind_swa_metadata(model)
        return partial(
            cls._init_dspark_swa_metadata,
            causal=causal,
            block_size=model.block_size,
        )

    def _init_dspark_swa_metadata(self, *, causal, block_size):
        if self.is_prefill or self.mtp_draft_input_hiddens is not None:
            # Context commits use causal target positions. Proposals attend to
            # history plus the complete block in the ring's future slots.
            return causal(self)

        pos = self.position_ids
        swa_width = ((self.swa_state.config.window_size + block_size + 127) // 128) * 128
        self.dsv4_swa_indices = torch.empty((pos.numel(), swa_width), dtype=torch.int32, device=pos.device)
        self.dsv4_swa_lengths = torch.empty_like(pos, dtype=torch.int32)
        self.dsv4_swa_write_slots = torch.empty_like(pos, dtype=torch.int32)
        self.dsv4_swa_runtime_write_slots = None
        build_dspark_swa_indices(
            token_req_idx=self.dsv4_sparse_req_idx,
            position_ids=self.position_ids,
            swa_state=self.swa_state,
            swa_indices=self.dsv4_swa_indices,
            swa_lengths=self.dsv4_swa_lengths,
            swa_write_slots=self.dsv4_swa_write_slots,
            block_size=block_size,
        )
