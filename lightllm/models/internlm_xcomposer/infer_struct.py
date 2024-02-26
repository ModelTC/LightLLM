import torch
import numpy as np
from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class InternlmComposerInferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None
        self.other_kv_index = None
    
    def init_some_extra_state(self, model, input_ids : torch.Tensor):
        super().init_some_extra_state(model, input_ids)
        if self.is_prefill:
            vocab_size = model.config['vocab_size']
            self.im_mask = (input_ids >= vocab_size)
            self.has_img = torch.sum(self.im_mask) > 0
            # print(self.im_mask.sum())
        else:
            self.im_mask = None
            self.has_img = False
        return
