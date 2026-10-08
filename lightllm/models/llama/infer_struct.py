from lightllm.common.basemodel import InferStateInfo


class LlamaInferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        self.rope = model.rope
        if self.is_prefill:
            self.max_seq_len = self.max_kv_seq_len
        self.position_cos, self.position_sin = self.rope.get_cos_sin(self.position_ids)
        return
