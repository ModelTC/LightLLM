import torch
from functools import partial
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.execution.runner import batch_metadata
from lightllm.common.execution.runner.batch_metadata import RotaryTables


class QwenInferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None
        self.logn_values = None

    @classmethod
    def bind_metadata(cls, model):
        if not model.config.get("use_dynamic_ntk", False):
            return super().bind_metadata(model)
        return partial(
            cls.prepare_metadata,
            tables=RotaryTables(model._cos_cached, model._sin_cached),
            seq_length=model.config.get("seq_length", 2048),
            max_ntk_alpha=model.max_ntk_alpha,
            logn_tensor=model.logn_tensor,
        )

    def prepare_metadata(self, *, tables, seq_length, max_ntk_alpha, logn_tensor):
        batch_metadata.prepare_sequence_metadata(self)
        if self.is_prefill:
            position_ids = self.position_ids
            self.position_sin = []
            self.position_cos = []
            infer_ntk_id = torch.clamp(
                torch.ceil(torch.log2(self.b_seq_len / seq_length) + 1),
                0,
                max_ntk_alpha,
            ).long()
            for i in range(len(infer_ntk_id)):
                _start = self.b1_cu_q_seq_len[i].item()
                _end = self.b1_cu_q_seq_len[i + 1].item()
                self.position_sin.append(tables.sin[infer_ntk_id[i]][position_ids[_start:_end]])
                self.position_cos.append(tables.cos[infer_ntk_id[i]][position_ids[_start:_end]])

            self.position_sin = torch.cat(self.position_sin, dim=0)
            self.position_cos = torch.cat(self.position_cos, dim=0)
            if logn_tensor is not None:
                self.logn_values = torch.index_select(logn_tensor, 0, position_ids).view(-1)
            position_ids = None
        else:
            infer_ntk_id = torch.clamp(
                torch.ceil(torch.log2(self.b_seq_len / seq_length) + 1),
                0,
                max_ntk_alpha,
            ).long()
            position_ids = self.position_ids
            self.position_cos = tables.cos[infer_ntk_id, position_ids].view(position_ids.shape[0], -1)
            self.position_sin = tables.sin[infer_ntk_id, position_ids].view(position_ids.shape[0], -1)
            if logn_tensor is not None:
                self.logn_values = torch.index_select(logn_tensor, 0, position_ids).view(-1)
            position_ids = None
        return
