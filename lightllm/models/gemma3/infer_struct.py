from lightllm.common.basemodel import InferStateInfo


class Gemma3InferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos_global = None
        self.position_sin_global = None
        self.position_sin_local = None
        self.position_cos_local = None

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        if self.is_prefill:
            self.max_seq_len = self.max_kv_seq_len
        self.position_cos_local, self.position_sin_local = model.rope_local.get_cos_sin(self.position_ids)
        self.position_cos_global, self.position_sin_global = model.rope_global.get_cos_sin(self.position_ids)
        self.position_cos, self.position_sin = self.position_cos_global, self.position_sin_global
        return
