import torch
from lightllm.models.deepseek2.flashattention_infer_struct import Deepseek2FlashAttentionStateInfo

class Deepseek3_2FlashAttentionStateInfo(Deepseek2FlashAttentionStateInfo):

    def __init__(self):
        super().__init__()
        self.lengths = None
        self.page_table_size_1 = None
        self.ks = None
        self.ke = None
        self.nsa_cu_seqlens_k = None
        self.index_topk = 2048
        return

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        super().init_some_extra_state(model, input_ids)
        # Ensure b_ready_cache_len is set for both prefill and decode modes
        if self.is_prefill:
            # b_ready_cache_len is already set in basemodel.py for prefill
            pass
        else:
            # In decode mode, b_ready_cache_len should be b_seq_len - b_q_seq_len
            # since b_q_seq_len represents the new tokens being processed
            if self.b_ready_cache_len is None:
                self.b_ready_cache_len = self.b_seq_len - self.b_q_seq_len
        
            self.nsa_cache_seqlens = self.b_att_seq_len.clamp(max=model.index_topk)
            assert self.nsa_cache_seqlens.dtype == torch.int32
            self.nsa_cu_seqlens_k =  torch.nn.functional.pad(
                torch.cumsum(self.nsa_cache_seqlens, dim=0, dtype=torch.int32), (1, 0)
            )   