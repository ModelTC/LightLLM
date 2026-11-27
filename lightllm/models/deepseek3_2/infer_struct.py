import torch

from lightllm.models.deepseek2.infer_struct import Deepseek2InferStateInfo
from lightllm.utils.dist_utils import get_current_device_id


class Deepseek3_2InferStateInfo(Deepseek2InferStateInfo):
    """Adds the NSA indexer's per-step metadata, shared by every layer's indexer.

    For each query row, indexer key positions ks..ke (exclusive) in a ragged per-request layout cover
    its visible keys, lengths = ke - ks, and nsa_ragged_mem_index maps ragged positions to cache slots.
    """

    def __init__(self):
        super().__init__()
        self.nsa_key_starts: torch.Tensor = None
        self.nsa_key_ends: torch.Tensor = None
        self.nsa_lengths: torch.Tensor = None
        self.nsa_ragged_mem_index: torch.Tensor = None

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        self._init_nsa_indexer_state(model)

    def _init_nsa_indexer_state(self, model):
        from lightllm.common.basemodel.triton_kernel.gen_nsa_ks_ke import gen_nsa_ks_ke

        if self.is_prefill:
            q_token_num = self.input_ids.shape[0]
            ragged_mem_index = None
        else:
            q_token_num = self.b_seq_len.shape[0]
            ragged_mem_index = None
            if self.batch_size <= model.graph_max_batch_size and self.max_kv_seq_len <= model.graph_max_len_in_batch:
                # Steps a CUDA graph may run write into storage the graph keeps, one buffer per microbatch.
                ragged_mem_index = _ragged_mem_buffer(model, self.microbatch_index)
        if ragged_mem_index is None:
            ragged_mem_index = torch.empty(self.total_token_num, dtype=torch.int32, device=get_current_device_id())
        self.nsa_ragged_mem_index = ragged_mem_index
        self.nsa_key_starts, self.nsa_key_ends, self.nsa_lengths = gen_nsa_ks_ke(
            b_seq_len=self.b_seq_len,
            b_q_seq_len=self.b_q_seq_len,
            b_req_idx=self.b_req_idx,
            req_to_token_index=self.req_manager.req_to_token_indexs,
            q_token_num=q_token_num,
            ragged_mem_index=ragged_mem_index,
            hold_req_idx=self.req_manager.HOLD_REQUEST_ID,
        )


def _ragged_mem_buffer(model, microbatch_index: int) -> torch.Tensor:
    buffers = getattr(model, "_nsa_ragged_mem_buffers", None)
    if buffers is None:
        size = model.graph_max_batch_size * model.max_seq_length
        buffers = [torch.empty(size, dtype=torch.int32, device=get_current_device_id()) for _ in range(2)]
        model._nsa_ragged_mem_buffers = buffers
    return buffers[microbatch_index]
