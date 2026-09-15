import torch
import triton

from lightllm.utils.vllm_utils import HAS_VLLM, vllm_ops

from .triton_kernel.index_quant import hadamard_transform_quant_fp8
from .triton_kernel.kpool import compress_pools, gather_pools, get_pool_ranges, expand_topk


class Glm5NextNsaInfer:
    """K-pool indexing with pooled token KV and a small per-request raw tail."""

    def __init__(self, layer_idx, network_config, tp_world_size):
        self.layer_idx = layer_idx
        self.topk = network_config["index_topk"]
        self.heads = network_config["index_n_heads"]
        self.dim = network_config["index_head_dim"]
        self.eps = network_config["rms_norm_eps"]

    def select_topk_indices(self, logits, lengths, indices):
        """Select row-relative indices, with valid entries before -1 padding."""
        if HAS_VLLM:
            # next_n=1 treats each query as an independent row, including prefill.
            # The decode entry splits long rows before merging their candidates.
            # Tested with vLLM 0.22.1; persistent_topk can drop candidates (#51782).
            vllm_ops.top_k_per_row_decode(
                logits, 1, lengths, indices, logits.shape[0], logits.stride(0), logits.stride(1), indices.shape[1]
            )
        else:
            positions = torch.arange(logits.shape[1], device=logits.device)
            logits.masked_fill_(positions[None, :] >= lengths[:, None], -float("inf"))
            selected = torch.topk(logits, indices.shape[1], dim=-1, sorted=True).indices
            indices.copy_(selected.masked_fill(selected >= lengths[:, None], -1))

    def _get_indices(self, hidden_states, q_lora, infer_state, att_state, layer_weight):
        k = layer_weight.k_norm_(layer_weight.wk_proj_.mm(hidden_states), eps=self.eps)
        gate = layer_weight.index_kpool_compress_gate.mm(hidden_states)
        raw = torch.cat((k, gate), -1)
        tail = infer_state.req_manager.get_indexer_tail_buffer(self.layer_idx)
        packed_buffer = infer_state.mem_manager.get_indexer_k_buffer(self.layer_idx)
        compress_pools(
            raw=raw,
            tail=tail,
            packed_buffer=packed_buffer,
            ape=layer_weight.index_kpool_compress_ape.weight,
            lengths=att_state.lengths,
            starts=att_state.ks,
            ragged=att_state.ragged_mem_index,
            req_idx=infer_state.b_req_idx,
            cu_q_lens=infer_state.b1_cu_q_seq_len,
            seq_lens=infer_state.b_seq_len,
            max_q_len=infer_state.max_q_seq_len,
            mtp_index=None if infer_state.is_prefill else infer_state.b_mtp_index,
        )

        if infer_state.max_kv_seq_len <= self.topk:
            return expand_topk(None, att_state.lengths, att_state.ks, att_state.ragged_mem_index, self.topk, dense=True)

        # The small indexer is replicated: no all-gather of query heads and
        # identical pool selection on every TP rank.
        q = layer_weight.wq_b_proj_.mm(q_lora).view(-1, self.heads, self.dim)
        q_fp8, q_scale = hadamard_transform_quant_fp8(q, scale=self.dim ** -0.5)
        weights = layer_weight.weights_proj_.mm(hidden_states.float())
        weights = weights * (self.heads ** -0.5 * self.dim ** -0.5) * q_scale.squeeze(-1)
        max_pools = triton.cdiv(infer_state.max_kv_seq_len, 4 * 128) * 128
        keys = gather_pools(
            packed_buffer,
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_req_idx,
            infer_state.b_seq_len,
            max_pools,
        )
        starts, ends, lengths = get_pool_ranges(
            att_state.lengths, infer_state.b1_cu_q_seq_len, infer_state.max_q_seq_len, max_pools
        )
        groups = torch.empty((q.shape[0], self.topk // 4), dtype=torch.int32, device=q.device)
        # Bound the transient score matrix independently of total batch length.
        chunk_size = max(1, min(q.shape[0], 16 * 1024 * 1024 // max_pools))
        import deep_gemm

        for start in range(0, q.shape[0], chunk_size):
            end = min(start + chunk_size, q.shape[0])
            logits = deep_gemm.fp8_mqa_logits(
                q_fp8[start:end],
                keys,
                weights[start:end],
                starts[start:end],
                ends[start:end],
                clean_logits=False,
                max_seqlen_k=max_pools,
            )
            self.select_topk_indices(logits, lengths[start:end], groups[start:end])
        return expand_topk(groups, att_state.lengths, att_state.ks, att_state.ragged_mem_index, self.topk)
