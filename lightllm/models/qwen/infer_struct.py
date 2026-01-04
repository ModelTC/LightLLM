import torch
import numpy as np
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel import InferStateInfo


class QwenInferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None
        self.logn_values = None

    def init_some_extra_state(self, model):
        use_dynamic_ntk = model.config.get("use_dynamic_ntk", False)
        if not use_dynamic_ntk:
            super().init_some_extra_state(model)
            return

        InferStateInfo.init_some_extra_state(self, model)
        if self.is_prefill:
            position_ids = self.position_ids
            self.position_sin = []
            self.position_cos = []
            infer_ntk_id = torch.clamp(
                torch.ceil(torch.log2(self.b_seq_len / model.config.get("seq_length", 2048)) + 1),
                0,
                model.max_ntk_alpha,
            ).long()
            for i in range(len(infer_ntk_id)):
                _start = self.b1_cu_q_seq_len[i].item()
                _end = self.b1_cu_q_seq_len[i + 1].item()
                self.position_sin.append(model._sin_cached[infer_ntk_id[i]][position_ids[_start:_end]])
                self.position_cos.append(model._cos_cached[infer_ntk_id[i]][position_ids[_start:_end]])

            self.position_sin = torch.cat(self.position_sin, dim=0)
            self.position_cos = torch.cat(self.position_cos, dim=0)
            if model.logn_tensor is not None:
                self.logn_values = torch.index_select(model.logn_tensor, 0, position_ids).view(-1)
            position_ids = None
        else:
            infer_ntk_id = torch.clamp(
                torch.ceil(torch.log2(self.b_seq_len / model.config.get("seq_length", 2048)) + 1),
                0,
                model.max_ntk_alpha,
            ).long()
            position_ids = self.position_ids
            self.position_cos = model._cos_cached[infer_ntk_id, position_ids].view(position_ids.shape[0], -1)
            self.position_sin = model._sin_cached[infer_ntk_id, position_ids].view(position_ids.shape[0], -1)
            if model.logn_tensor is not None:
                self.logn_values = torch.index_select(model.logn_tensor, 0, position_ids).view(-1)
            position_ids = None
        return
