from typing import NamedTuple

import torch

from lightllm.common.basemodel.infer_struct import InferStateInfo


class SparseQueryMetadata(NamedTuple):
    q_kv_seq_len: torch.Tensor
    q_req_idx: torch.Tensor


class Glm5NextInferStateInfo(InferStateInfo):
    graph_input_names = InferStateInfo.graph_input_names + (
        "q_kv_seq_len",
        "q_req_idx",
    )

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        self.q_kv_seq_len = self.position_ids + 1 if self.is_prefill else self.b_seq_len
        self.q_req_idx = (
            torch.repeat_interleave(self.b_req_idx, self.b_q_seq_len, output_size=self.input_ids.shape[0])
            if self.is_prefill
            else self.b_req_idx
        )

    @property
    def sparse_queries(self):
        return SparseQueryMetadata(self.q_kv_seq_len, self.q_req_idx)
