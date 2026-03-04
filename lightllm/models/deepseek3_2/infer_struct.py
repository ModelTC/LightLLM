import torch
import weakref
from lightllm.models.deepseek2.infer_struct import Deepseek2InferStateInfo
from lightllm.models.deepseek3_2.mem_manager import Deepseek3_2MemoryManager


class Deepseek3_2InferStateInfo(Deepseek2InferStateInfo):
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
                and self.max_kv_seq_len <= model.graph_max_len_in_batch
            ):
                return True
        return False

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)

        self._model_ref = weakref.ref(model)

        assert isinstance(self.mem_manager, Deepseek3_2MemoryManager)
        self.indexer_ks_buffer = self.mem_manager.indexer_ks_buffer

        if self.is_prefill:
            self._init_nsa_indexing_prefill()
        else:
            if self.b_ready_cache_len is None:
                self.b_ready_cache_len = torch.zeros_like(self.b_seq_len)

            use_cuda_graph_buffers = self._check_use_cuda_graph_buffers()
            buffer = None

            if use_cuda_graph_buffers:
                buffers = self.get_nsa_buffers(model.graph_max_batch_size, model.graph_max_len_in_batch)
                buffer = buffers[self.microbatch_index]
                self.nsa_cache_seqlens = buffer["nsa_cache_seqlens"][: self.batch_size]
                self.nsa_cu_seqlens_k = buffer["nsa_cu_seqlens_k"][: self.batch_size + 1]
            else:
                self.nsa_cache_seqlens = torch.empty(self.batch_size, dtype=torch.int32, device="cuda")
                self.nsa_cu_seqlens_k = torch.empty(self.batch_size + 1, dtype=torch.int32, device="cuda")

            self.nsa_cache_seqlens.copy_(self.b_kv_seq_len.clamp(max=self.index_topk))
            assert self.nsa_cache_seqlens.dtype == torch.int32

            torch.cumsum(self.nsa_cache_seqlens, dim=0, dtype=torch.int32, out=self.nsa_cu_seqlens_k[1:])
            self.nsa_cu_seqlens_k[0] = 0

            self._init_nsa_indexing_decode(use_cuda_graph_buffers, buffer)

    def _init_nsa_indexing_decode(self, use_cuda_graph_buffers, buffer):
        """Optimized NSA indexing for decode: b_q_seq_len=1 per request.

        In decode, each request generates exactly 1 token, so:
        - total_q_len = batch_size (no .item() needed)
        - ks[i] = cumsum_offset[i], ke[i] = cumsum_offset[i] + 1
        - lengths[i] = b_seq_len[i]
        - No repeat_interleave, no token_in_req math needed.
        """
        b_seq_len = self.b_seq_len
        b_req_idx = self.b_req_idx
        num_seq = self.batch_size

        # Cumulative seq_len offsets for ks/ke: [0, s0, s0+s1, ...]
        cum_seq = torch.cumsum(b_seq_len, dim=0, dtype=torch.int32)

        if use_cuda_graph_buffers:
            model = self._model_ref()
            max_seq_len = model.graph_max_len_in_batch

            # ks, ke, lengths — write directly into buffer slices
            buf_ks = buffer["ks"][:num_seq]
            buf_ke = buffer["ke"][:num_seq]
            buf_lengths = buffer["lengths"][:num_seq]

            # ks[0] = 0, ks[i] = cum_seq[i-1]
            buf_ks[0] = 0
            if num_seq > 1:
                buf_ks[1:].copy_(cum_seq[: num_seq - 1])
            # ke = ks + 1
            torch.add(buf_ks, 1, out=buf_ke)
            # lengths = b_seq_len
            buf_lengths.copy_(b_seq_len.int())

            self.ks = buf_ks
            self.ke = buf_ke
            self.lengths = buf_lengths

            # page_table: zero buffer slice, then fill valid entries
            page_table = buffer["page_table_size_1"][:num_seq, :max_seq_len]
            page_table.zero_()
            all_rows = self.req_manager.req_to_token_indexs[b_req_idx, :max_seq_len]
            seq_range = torch.arange(max_seq_len, device=b_seq_len.device)
            valid_mask = seq_range.unsqueeze(0) < b_seq_len.unsqueeze(1)
            page_table[valid_mask] = all_rows[valid_mask].int()
            self.page_table_size_1 = page_table

            # req_all_mem_index: use padded [num_seq * max_seq_len] layout
            # Downstream uses ks/ke masking so padded entries are safe
            max_total_seq = num_seq * max_seq_len
            buf_mem = buffer["req_all_mem_index"][:max_total_seq]
            buf_mem.copy_(all_rows.reshape(-1))
            self.req_all_mem_index = buf_mem
        else:
            # Non-CUDA-graph decode: simplified formulas, fresh tensors
            max_seq_len = b_seq_len.max().item()

            # ks/ke/lengths
            seq_offsets = torch.empty_like(cum_seq)
            seq_offsets[0] = 0
            if num_seq > 1:
                seq_offsets[1:] = cum_seq[:-1]

            self.ks = seq_offsets
            self.ke = (seq_offsets + 1).int()
            self.lengths = b_seq_len.int()

            # page_table and req_all_mem_index
            all_rows = self.req_manager.req_to_token_indexs[b_req_idx, :max_seq_len]
            seq_range = torch.arange(max_seq_len, device=b_seq_len.device)
            valid_mask = seq_range.unsqueeze(0) < b_seq_len.unsqueeze(1)

            page_table = torch.zeros((num_seq, max_seq_len), dtype=torch.int, device=b_seq_len.device)
            page_table[valid_mask] = all_rows[valid_mask].int()
            self.page_table_size_1 = page_table

            self.req_all_mem_index = all_rows[valid_mask]

    def _init_nsa_indexing_prefill(self):
        """NSA indexing for prefill: variable q lengths, generic vectorized path."""
        b_seq_len = self.b_seq_len
        b_q_seq_len = self.b_q_seq_len
        b_req_idx = self.b_req_idx
        num_seq = b_req_idx.shape[0]
        device = b_seq_len.device

        max_seq_len = b_seq_len.max().item()
        total_q_len = b_q_seq_len.sum().item()

        # page_table_size_1 and req_all_mem_index
        all_rows = self.req_manager.req_to_token_indexs[b_req_idx, :max_seq_len]
        seq_range = torch.arange(max_seq_len, device=device)
        valid_mask = seq_range.unsqueeze(0) < b_seq_len.unsqueeze(1)

        page_table = torch.zeros((num_seq, max_seq_len), dtype=torch.int, device=device)
        page_table[valid_mask] = all_rows[valid_mask].int()
        self.page_table_size_1 = page_table
        self.req_all_mem_index = all_rows[valid_mask]

        # ks, ke, lengths — generic vectorized for variable q lengths
        cum_seq = torch.cumsum(b_seq_len, dim=0)
        seq_offsets = torch.zeros_like(cum_seq)
        seq_offsets[1:] = cum_seq[:-1]

        req_indices = torch.repeat_interleave(torch.arange(num_seq, device=device), b_q_seq_len)

        cum_q = torch.cumsum(b_q_seq_len, dim=0)
        q_offsets = torch.zeros_like(cum_q)
        q_offsets[1:] = cum_q[:-1]
        token_in_req = torch.arange(total_q_len, device=device) - q_offsets[req_indices]

        self.ks = seq_offsets[req_indices].int()
        self.ke = (seq_offsets[req_indices] + token_in_req + 1).int()
        self.lengths = (b_seq_len[req_indices] - b_q_seq_len[req_indices] + token_in_req + 1).int()
