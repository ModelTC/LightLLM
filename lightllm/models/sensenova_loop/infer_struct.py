import torch

from lightllm.models.qwen3_5.infer_struct import Qwen35InferStateInfo
from .dsa import query_metadata
from .dsa_graph import score_bucket


class DsaLoopInferStateInfo(Qwen35InferStateInfo):
    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        if not model.dsa:
            return
        capacity = self.req_manager.req_to_token_indexs.shape[1]
        if not self.is_prefill and (self.is_cuda_graph_capture or self.is_cuda_graph_replay):
            capacity = min(capacity, model.graph.graph_max_len_in_batch)
        self.dsa_score_max_keys = score_bucket(self.max_kv_seq_len, capacity)
        # These tensors are shared by all execution layers. Native graph replay
        # copies them with the other infer-state tensor fields.
        self.dsa_query_requests, self.dsa_query_positions = query_metadata(
            self.b1_cu_q_seq_len, self.b_kv_seq_len, self.b_req_idx, self.input_ids.shape[0]
        )
        self.dsa_query_lengths = (self.dsa_query_positions + 1).clamp(0, self.dsa_score_max_keys).to(torch.int32)
        rows = self.input_ids.shape[0]
        cu_q = getattr(model, "_dsa_cu_q", None)
        if cu_q is None or cu_q.numel() < rows + 1:
            capacity = max(model.batch_max_tokens or 0, model.graph_max_batch_size, rows)
            cu_q = torch.arange(capacity + 1, dtype=torch.int32, device=self.input_ids.device)
            model._dsa_cu_q = cu_q
        self.dsa_cu_q = cu_q[: rows + 1]
