from lightllm.models.deepseek_v4.infer_struct import DeepseekV4InferStateInfo


class DeepseekV41InferStateInfo(DeepseekV4InferStateInfo):
    bidirectional_image_attention = False

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        self.v41_sparse_selection = None
        self.candidate_block_mask = None
        self.engram_hashes = model.engram_hasher(self)
