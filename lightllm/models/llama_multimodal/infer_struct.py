import torch
from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class LlamaMultiModalInferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
    
    def init_some_extra_state(self, 
            model, 
            batch_size, 
            total_token_num,
            max_len_in_batch,
            input_ids : torch.Tensor,
            b_loc : torch.Tensor,
            b_start_loc : torch.Tensor,
            b_seq_len : torch.Tensor,
            is_prefill,
            **kwargs):
        super().init_some_extra_state(model, batch_size, total_token_num, max_len_in_batch, input_ids, b_loc, b_start_loc, b_seq_len, is_prefill)
        self.kwargs = kwargs
