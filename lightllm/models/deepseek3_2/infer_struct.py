import torch
import weakref
from lightllm.models.deepseek2.infer_struct import Deepseek2InferStateInfo
from lightllm.models.deepseek3_2.mem_manager import Deepseek3_2MemoryManager


class Deepseek3_2FlashAttentionStateInfo(Deepseek2InferStateInfo):
    _shared_nsa_buffers = None

    def __init__(self):
        super().__init__()
        self.lengths = None
        self.page_table_size_1 = None
        self.ks = None
        self.ke = None
        self.nsa_cu_seqlens_k = None
        self.index_topk = 2048
        return

    @classmethod
    def get_nsa_buffers(cls, graph_max_batch_size: int, max_seq_len: int):
        """Get or create pre-allocated buffers for CUDA graph execution"""
        if cls._shared_nsa_buffers is None:
            max_total_q_tokens = graph_max_batch_size * max_seq_len
            max_total_tokens = graph_max_batch_size * max_seq_len

            cls._shared_nsa_buffers = [
                {
                    "ks": torch.empty(max_total_q_tokens, dtype=torch.int, device="cuda"),
                    "ke": torch.empty(max_total_q_tokens, dtype=torch.int, device="cuda"),
                    "lengths": torch.empty(max_total_q_tokens, dtype=torch.int, device="cuda"),
                    "page_table_size_1": torch.empty(graph_max_batch_size, max_seq_len, dtype=torch.int, device="cuda"),
                    "req_all_mem_index": torch.empty(max_total_tokens, dtype=torch.int64, device="cuda"),
                    "nsa_cache_seqlens": torch.empty(graph_max_batch_size, dtype=torch.int32, device="cuda"),
                    "nsa_cu_seqlens_k": torch.empty(graph_max_batch_size + 1, dtype=torch.int32, device="cuda"),
                },
                {
                    "ks": torch.empty(max_total_q_tokens, dtype=torch.int, device="cuda"),
                    "ke": torch.empty(max_total_q_tokens, dtype=torch.int, device="cuda"),
                    "lengths": torch.empty(max_total_q_tokens, dtype=torch.int, device="cuda"),
                    "page_table_size_1": torch.empty(graph_max_batch_size, max_seq_len, dtype=torch.int, device="cuda"),
                    "req_all_mem_index": torch.empty(max_total_tokens, dtype=torch.int64, device="cuda"),
                    "nsa_cache_seqlens": torch.empty(graph_max_batch_size, dtype=torch.int32, device="cuda"),
                    "nsa_cu_seqlens_k": torch.empty(graph_max_batch_size + 1, dtype=torch.int32, device="cuda"),
                },
            ]
        return cls._shared_nsa_buffers

    def _check_use_cuda_graph_buffers(self):
        if hasattr(self, "_model_ref"):
            model = self._model_ref()
            if (
                model is not None
                and hasattr(model, "graph_max_batch_size")
                and hasattr(model, "graph_max_len_in_batch")
                and self.batch_size <= model.graph_max_batch_size
                and self.max_len_in_batch <= model.graph_max_len_in_batch
            ):
                return True
        return False

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)

        self._model_ref = weakref.ref(model)

        assert isinstance(self.mem_manager, Deepseek3_2MemoryManager)
        self.indexer_ks_buffer = self.mem_manager.indexer_ks_buffer

        if self.is_prefill:
            pass
        else:
            if self.b_ready_cache_len is None:
                self.b_ready_cache_len = torch.zeros_like(self.b_seq_len)

            use_cuda_graph_buffers = self._check_use_cuda_graph_buffers()

            if use_cuda_graph_buffers:
                buffers = self.get_nsa_buffers(model.graph_max_batch_size, model.graph_max_len_in_batch)
                buffer = buffers[self.microbatch_index]
                self.nsa_cache_seqlens = buffer["nsa_cache_seqlens"][: self.batch_size]
                self.nsa_cu_seqlens_k = buffer["nsa_cu_seqlens_k"][: self.batch_size + 1]
            else:
                self.nsa_cache_seqlens = torch.empty(self.batch_size, dtype=torch.int32, device="cuda")
                self.nsa_cu_seqlens_k = torch.empty(self.batch_size + 1, dtype=torch.int32, device="cuda")

            self.nsa_cache_seqlens.copy_(self.b_att_seq_len.clamp(max=self.index_topk))
            assert self.nsa_cache_seqlens.dtype == torch.int32

            torch.cumsum(self.nsa_cache_seqlens, dim=0, dtype=torch.int32, out=self.nsa_cu_seqlens_k[1:])
            self.nsa_cu_seqlens_k[0] = 0

        self._init_nsa_indexing_structures()

    def _init_nsa_indexing_structures(self):
        """Pre-compute ks, ke, lengths, and page_table_size_1 for NSA indexer.

        Fully vectorized: eliminates per-request .item() CPU-GPU syncs.
        """
        b_seq_len = self.b_seq_len
        b_q_seq_len = self.b_q_seq_len
        b_req_idx = self.b_req_idx
        num_seq = b_req_idx.shape[0]
        device = b_seq_len.device

        # Only 3 scalar syncs needed (for tensor shapes)
        max_seq_len = b_seq_len.max().item()
        total_q_len = b_q_seq_len.sum().item()
        total_seq_len = b_seq_len.sum().item()

        # --- page_table_size_1 and req_all_mem_index (vectorized gather) ---
        all_rows = self.req_manager.req_to_token_indexs[b_req_idx, :max_seq_len]
        seq_range = torch.arange(max_seq_len, device=device)
        valid_mask = seq_range.unsqueeze(0) < b_seq_len.unsqueeze(1)

        # page_table_size_1: [batch, max_seq_len] zero-padded memory indices
        page_table = torch.zeros((num_seq, max_seq_len), dtype=torch.int, device=device)
        page_table[valid_mask] = all_rows[valid_mask].int()

        # req_all_mem_index: flattened valid memory indices across all requests
        req_all_mem_index = all_rows[valid_mask]

        # --- ks, ke, lengths (vectorized computation) ---
        # Cumulative seq_len offsets: [0, seq_len[0], seq_len[0]+seq_len[1], ...]
        cum_seq = torch.cumsum(b_seq_len, dim=0)
        seq_offsets = torch.zeros_like(cum_seq)
        seq_offsets[1:] = cum_seq[:-1]

        # Expand per-request values to per-token using repeat_interleave
        req_indices = torch.repeat_interleave(torch.arange(num_seq, device=device), b_q_seq_len)

        # Token position within each request's q_seq
        cum_q = torch.cumsum(b_q_seq_len, dim=0)
        q_offsets = torch.zeros_like(cum_q)
        q_offsets[1:] = cum_q[:-1]
        token_in_req = torch.arange(total_q_len, device=device) - q_offsets[req_indices]

        # ks[t] = seq_offset of request owning token t
        # ke[t] = seq_offset + position_in_q + 1
        # lengths[t] = seq_len - q_seq_len + position_in_q + 1
        ks = seq_offsets[req_indices].int()
        ke = (seq_offsets[req_indices] + token_in_req + 1).int()
        lengths = (b_seq_len[req_indices] - b_q_seq_len[req_indices] + token_in_req + 1).int()

        # --- Assign results (CUDA graph buffer or new tensors) ---
        use_cuda_graph_buffers = self._check_use_cuda_graph_buffers()

        if use_cuda_graph_buffers:
            model = self._model_ref()
            buffers = self.get_nsa_buffers(model.graph_max_batch_size, model.graph_max_len_in_batch)
            buffer = buffers[self.microbatch_index]

            self.ks = buffer["ks"][:total_q_len]
            self.ke = buffer["ke"][:total_q_len]
            self.lengths = buffer["lengths"][:total_q_len]
            self.page_table_size_1 = buffer["page_table_size_1"][:num_seq, :max_seq_len]
            self.req_all_mem_index = buffer["req_all_mem_index"][:total_seq_len]

            self.ks.copy_(ks)
            self.ke.copy_(ke)
            self.lengths.copy_(lengths)
            self.page_table_size_1.copy_(page_table)
            self.req_all_mem_index.copy_(req_all_mem_index)
        else:
            self.ks = ks
            self.ke = ke
            self.lengths = lengths
            self.page_table_size_1 = page_table
            self.req_all_mem_index = req_all_mem_index
