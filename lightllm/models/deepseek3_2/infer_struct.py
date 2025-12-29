import torch
import weakref
from lightllm.models.deepseek2.flashattention_infer_struct import Deepseek2FlashAttentionStateInfo
from lightllm.models.deepseek3_2.mem_manager import Deepseek3_2MemoryManager


class Deepseek3_2FlashAttentionStateInfo(Deepseek2FlashAttentionStateInfo):
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
            # Pre-allocate buffers for max possible sizes
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
                {  # Second buffer for microbatch overlap if needed
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

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        super().init_some_extra_state(model, input_ids)

        # Store weak reference to model for accessing graph parameters
        self._model_ref = weakref.ref(model)

        assert isinstance(self.mem_manager, Deepseek3_2MemoryManager)
        self.indexer_ks_mem_manager = self.mem_manager.indexer_ks_mem_manager

        # Ensure b_ready_cache_len is set for both prefill and decode modes
        if self.is_prefill:
            # b_ready_cache_len is already set in basemodel.py for prefill
            pass
        else:
            # In decode mode, b_ready_cache_len is set by the router/scheduler
            # based on actual prefix cache hits. If it's None (no prefix cache enabled),
            # it should be 0, not computed from b_seq_len - b_q_seq_len
            if self.b_ready_cache_len is None:
                self.b_ready_cache_len = torch.zeros_like(self.b_seq_len)

            # Check if we can use CUDA graph based on batch size and max_len constraints
            use_cuda_graph_buffers = False
            if (
                hasattr(model, "graph_max_batch_size")
                and hasattr(model, "graph_max_len_in_batch")
                and self.batch_size <= model.graph_max_batch_size
                and self.max_len_in_batch <= model.graph_max_len_in_batch
            ):
                use_cuda_graph_buffers = True

            # Setup nsa_cache_seqlens and nsa_cu_seqlens_k with pre-allocated buffers if using CUDA graph
            if use_cuda_graph_buffers:
                buffers = self.get_nsa_buffers(model.graph_max_batch_size, model.graph_max_len_in_batch)
                buffer = buffers[self.microbatch_index]

                # Use views into pre-allocated buffers
                self.nsa_cache_seqlens = buffer["nsa_cache_seqlens"][: self.batch_size]
                self.nsa_cu_seqlens_k = buffer["nsa_cu_seqlens_k"][: self.batch_size + 1]
            else:
                # Create new tensors dynamically
                self.nsa_cache_seqlens = torch.empty(self.batch_size, dtype=torch.int32, device="cuda")
                self.nsa_cu_seqlens_k = torch.empty(self.batch_size + 1, dtype=torch.int32, device="cuda")

            # Calculate actual values
            self.nsa_cache_seqlens.copy_(self.b_att_seq_len.clamp(max=self.index_topk))
            assert self.nsa_cache_seqlens.dtype == torch.int32

            # Compute cumulative sum with padding
            torch.cumsum(self.nsa_cache_seqlens, dim=0, dtype=torch.int32, out=self.nsa_cu_seqlens_k[1:])
            self.nsa_cu_seqlens_k[0] = 0

        # Pre-compute NSA indexer indexing structures
        self._init_nsa_indexing_structures()

    def _init_nsa_indexing_structures(self):
        """Pre-compute ks, ke, lengths, and page_table_size_1 for NSA indexer"""
        req_all_mem_index_list = []
        ks_list = []
        ke_list = []
        lengths_list = []
        offset = 0
        num_seq_len = self.b_req_idx.shape[0]
        max_seq_len = self.b_seq_len.max().item()

        # Calculate total sizes needed
        total_q_len = sum(self.b_q_seq_len[i].item() for i in range(num_seq_len))
        total_seq_len = sum(self.b_seq_len[i].item() for i in range(num_seq_len))

        # Check if we should use CUDA graph buffers
        use_cuda_graph_buffers = False
        if hasattr(self, "_model_ref"):
            model = self._model_ref()
            if (
                model is not None
                and hasattr(model, "graph_max_batch_size")
                and hasattr(model, "graph_max_len_in_batch")
                and self.batch_size <= model.graph_max_batch_size
                and self.max_len_in_batch <= model.graph_max_len_in_batch
            ):
                use_cuda_graph_buffers = True

        if use_cuda_graph_buffers:
            # Use pre-allocated buffers for CUDA graph
            model = self._model_ref()
            buffers = self.get_nsa_buffers(model.graph_max_batch_size, model.graph_max_len_in_batch)
            buffer = buffers[self.microbatch_index]

            # Use views into pre-allocated buffers
            self.ks = buffer["ks"][:total_q_len]
            self.ke = buffer["ke"][:total_q_len]
            self.lengths = buffer["lengths"][:total_q_len]
            self.page_table_size_1 = buffer["page_table_size_1"][:num_seq_len, :max_seq_len]
            self.req_all_mem_index = buffer["req_all_mem_index"][:total_seq_len]

            # Zero out page_table_size_1 before filling
            self.page_table_size_1.zero_()

            # Compute and copy values into the pre-allocated buffer views
            ks_offset = 0
            ke_offset = 0
            lengths_offset = 0
            req_offset = 0
            seq_offset = 0

            for i in range(num_seq_len):
                seq_len = self.b_seq_len[i].item()
                q_seq_len = self.b_q_seq_len[i].item()
                req_idx = self.b_req_idx[i].item()
                mem_index = self.req_manager.req_to_token_indexs[req_idx, :seq_len]

                # Copy req_all_mem_index
                self.req_all_mem_index[req_offset : req_offset + seq_len] = mem_index

                # Fill page_table_size_1
                self.page_table_size_1[i, :seq_len] = mem_index

                # Fill ks, ke, lengths
                self.ks[ks_offset : ks_offset + q_seq_len].fill_(seq_offset)
                self.ke[ke_offset : ke_offset + q_seq_len] = torch.arange(
                    seq_offset + 1, seq_offset + q_seq_len + 1, dtype=torch.int, device="cuda"
                )
                self.lengths[lengths_offset : lengths_offset + q_seq_len] = torch.arange(
                    seq_len - q_seq_len + 1, seq_len + 1, dtype=torch.int, device="cuda"
                )

                ks_offset += q_seq_len
                ke_offset += q_seq_len
                lengths_offset += q_seq_len
                req_offset += seq_len
                seq_offset += seq_len
        else:
            # Original dynamic allocation for non-CUDA graph mode
            self.page_table_size_1 = torch.zeros((num_seq_len, max_seq_len), dtype=torch.int, device="cuda")

            for i in range(num_seq_len):
                seq_len = self.b_seq_len[i].item()
                q_seq_len = self.b_q_seq_len[i].item()
                req_idx = self.b_req_idx[i].item()
                mem_index = self.req_manager.req_to_token_indexs[req_idx, :seq_len]
                req_all_mem_index_list.append(mem_index)
                self.page_table_size_1[i, :seq_len] = mem_index
                ks = torch.zeros(q_seq_len, dtype=torch.int, device="cuda") + offset
                ke = torch.arange(q_seq_len, dtype=torch.int, device="cuda") + offset + 1
                ks_list.append(ks)
                ke_list.append(ke)
                lengths_list.append(torch.arange(seq_len - q_seq_len + 1, seq_len + 1, dtype=torch.int, device="cuda"))
                offset += seq_len

            self.req_all_mem_index = torch.cat(req_all_mem_index_list, dim=0)
            self.ks = torch.cat(ks_list, dim=0)
            self.ke = torch.cat(ke_list, dim=0)
            self.lengths = torch.cat(lengths_list, dim=0)
