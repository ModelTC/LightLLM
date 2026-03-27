import torch
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.utils.envs_utils import get_env_start_args


class Qwen3NextInferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self.gate_value = None

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        self.b_att_seq_len = self.b_seq_len
        self.b_buffer_idx = model.req_manager.req_to_buffer_index[self.b_req_idx, 0].contiguous()

        return
