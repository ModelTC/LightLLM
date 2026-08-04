import torch

from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class Qwen3DFlashInferStateInfo(LlamaInferStateInfo):
    """DFlash metadata on top of the normal prefill state."""

    def __init__(self):
        super().__init__()
        self.prefill_causal: bool = True
        self.decode_causal: bool = True
        self.decode_mtp_step: int = None

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        if self.is_prefill:
            self.prefill_causal = False
        else:
            self.decode_causal = False
            self.decode_mtp_step = model.block_size - 1
            self.is_draft_model = True
        return

    @staticmethod
    def build_draft_query_position_ids(
        *,
        selected_seq_len: torch.Tensor,
        b_position_delta: torch.Tensor = None,
        draft_step: int,
    ) -> torch.Tensor:
        offsets = torch.arange(
            int(draft_step),
            dtype=torch.long,
            device=selected_seq_len.device,
        )
        position_ids = selected_seq_len.to(torch.long).view(-1, 1) + offsets.view(1, -1)
        if b_position_delta is not None:
            position_ids = position_ids + b_position_delta.to(torch.long).view(-1, 1)
        return position_ids
