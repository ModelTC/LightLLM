import torch
from lightllm.models.deepseek2.flashattention_infer_struct import Deepseek2FlashAttentionStateInfo
from lightllm.models.deepseek3_2.mem_manager import Deepseek3_2MemoryManager

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
        assert isinstance(self.mem_manager, Deepseek3_2MemoryManager)
        self.indexer_ks_mem_manager = self.mem_manager.indexer_ks_mem_manager

        # Ensure b_ready_cache_len is set for both prefill and decode modes
        if self.is_prefill:
            # b_ready_cache_len is already set in basemodel.py for prefill
            pass
        else:
            # In decode mode, b_ready_cache_len should be b_seq_len - b_q_seq_len
            # since b_q_seq_len represents the new tokens being processed
            if self.b_ready_cache_len is None:
                self.b_ready_cache_len = self.b_seq_len - self.b_q_seq_len

            self.nsa_cache_seqlens = self.b_att_seq_len.clamp(max=self.index_topk)
            assert self.nsa_cache_seqlens.dtype == torch.int32
            self.nsa_cu_seqlens_k =  torch.nn.functional.pad(
                torch.cumsum(self.nsa_cache_seqlens, dim=0, dtype=torch.int32), (1, 0)
            )

        # Pre-compute NSA indexer indexing structures
        self._init_nsa_indexing_structures()

    def _init_nsa_indexing_structures(self):
        """Pre-compute ks, ke, lengths, and page_table_size_1 for NSA indexer"""
        mem_index_list = []
        ks_list = []
        ke_list = []
        lengths_list = []
        offset = 0
        num_seq_len = self.b_req_idx.shape[0]
        self.page_table_size_1 = torch.zeros((num_seq_len, self.b_seq_len.max()), dtype=torch.int, device='cuda')

        for i in range(num_seq_len):
            seq_len = self.b_seq_len[i]
            q_seq_len = self.b_q_seq_len[i]
            mem_index = self.req_manager.req_to_token_indexs[i, :seq_len]
            mem_index_list.append(mem_index)
            self.page_table_size_1[i, :seq_len] = mem_index
            ks = torch.zeros(q_seq_len, dtype=torch.int, device='cuda') + offset
            ke = torch.arange(q_seq_len, dtype=torch.int, device='cuda') + offset + 1
            ks_list.append(ks)
            ke_list.append(ke)
            lengths_list.append(torch.arange(seq_len - q_seq_len + 1, seq_len + 1, dtype=torch.int, device='cuda'))
            offset += seq_len

        self.mem_index = torch.cat(mem_index_list, dim=0)
        self.ks = torch.cat(ks_list, dim=0)
        self.ke = torch.cat(ke_list, dim=0)
        self.lengths = torch.cat(lengths_list, dim=0)