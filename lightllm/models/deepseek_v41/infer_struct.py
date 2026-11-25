import torch

from lightllm.models.deepseek_v4.infer_struct import DeepseekV4InferStateInfo


class DeepseekV41InferStateInfo(DeepseekV4InferStateInfo):
    bidirectional_image_attention = False

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        self.v41_sparse_selection = None
        self.candidate_block_mask = None
        self.engram_hashes = model.engram_hasher(self)

    def _indexer_valid_lengths(self, ratio):
        valid = ((self.position_ids + 1) // ratio).to(torch.int32)
        return valid.masked_fill_(self.dsv4_sparse_req_idx == self.req_manager.HOLD_REQUEST_ID, 0)
