import torch
import numpy as np
from lightllm.common.basemodel import InferStateInfo

class QwenInferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None
        self.other_kv_index = None
        self.logn_values = None

    def init_some_extra_state(self, model, input_ids : torch.Tensor):
        if self.is_prefill:
            b_seq_len_numpy = self.b_seq_len.cpu().numpy()
            position_ids = torch.from_numpy(np.concatenate([np.arange(0, b_seq_len_numpy[i])
                                            for i in range(len(b_seq_len_numpy))], axis=0)).cuda()
            infer_ntk_alphas = np.maximum(2 ** (np.ceil(np.log2(b_seq_len_numpy / model.config.get("seq_length", 2048)) + 1)), 1)
            model.infer_ntk_id = torch.tensor([model._ntk_to_id[i] for i in infer_ntk_alphas]).cuda()
            self.position_sin = []
            self.position_cos = []
            model.infer_ntk_id = [model._ntk_to_id[i] for i in infer_ntk_alphas]
            for i in range(len(infer_ntk_alphas)):
                self.position_sin.append(model._sin_cached[model.infer_ntk_id[i]][position_ids[self.b_start_loc[i]: self.b_start_loc[i] + self.b_seq_len[i]]])
                self.position_cos.append(model._cos_cached[model.infer_ntk_id[i]][position_ids[self.b_start_loc[i]: self.b_start_loc[i] + self.b_seq_len[i]]])

            self.position_sin = torch.cat(self.position_sin, dim=0)
            self.position_cos = torch.cat(self.position_cos, dim=0)
            if model.logn_tensor is not None:
                self.logn_values = torch.index_select(model.logn_tensor, 0, position_ids).view(-1)
            position_ids = None
        else:
            position_ids = self.b_seq_len - 1
            self.position_cos = model._cos_cached[model.infer_ntk_id, position_ids].view(position_ids.shape[0], -1)
            self.position_sin = model._sin_cached[model.infer_ntk_id, position_ids].view(position_ids.shape[0], -1)
            if model.logn_tensor is not None:
                self.logn_values = torch.index_select(model.logn_tensor, 0, position_ids).view(-1)
            self.other_kv_index = self.req_manager.req_to_token_indexs[self.b_req_idx[0], 0].item()
            position_ids = None
        return
