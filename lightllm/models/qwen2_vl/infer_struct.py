import torch
import numpy as np
from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class Qwen2VLInferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        if self.is_prefill:
            b_seq_len_numpy = self.b_seq_len.cpu().numpy()
            self.max_seq_len = b_seq_len_numpy.max()
            b_ready_cache_len_numpy = self.b_ready_cache_len.cpu().numpy()
            position_ids = torch.from_numpy(
                np.concatenate(
                    [np.arange(b_ready_cache_len_numpy[i], b_seq_len_numpy[i]) for i in range(len(b_seq_len_numpy))]
                )
            ).cuda()
            self.position_sin = model._sin_cached[:, position_ids, :].unsqueeze(1)
            self.position_cos = model._cos_cached[:, position_ids, :].unsqueeze(1)
            position_ids = None
        else:
            position_ids = self.b_seq_len - 1
            self.position_sin = model._sin_cached[:, position_ids, :].unsqueeze(1)
            self.position_cos = model._cos_cached[:, position_ids, :].unsqueeze(1)
        return
