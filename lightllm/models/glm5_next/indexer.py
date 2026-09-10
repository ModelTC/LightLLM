import torch
import triton

from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv
from .triton_kernel.index_quant import hadamard_transform_quant_fp8
from .triton_kernel.kpool import compress_pools, gather_pools, expand_topk


class Glm5NextNsaInfer:
    """K-pool indexing with all persistent history stored in token KV."""

    def __init__(self, layer_idx, network_config, tp_world_size):
        self.layer_idx = layer_idx
        self.topk = network_config["index_topk"]
        self.heads = network_config["index_n_heads"]
        self.dim = network_config["index_head_dim"]
        self.eps = network_config["rms_norm_eps"]

    def _get_indices(self, hidden_states, q_lora, infer_state, att_state, layer_weight):
        k = layer_weight.k_norm_(layer_weight.wk_proj_.mm(hidden_states), eps=self.eps)
        gate = layer_weight.index_kpool_compress_gate.mm(hidden_states)
        raw = torch.cat((k, gate), -1).unsqueeze(1)
        raw_buffer = infer_state.mem_manager.get_indexer_raw_buffer(self.layer_idx)
        packed_buffer = infer_state.mem_manager.get_indexer_k_buffer(self.layer_idx)
        destindex_copy_kv(raw, infer_state.mem_index, raw_buffer)
        compress_pools(
            raw_buffer,
            packed_buffer,
            layer_weight.index_kpool_compress_ape.weight,
            att_state.lengths,
            att_state.ks,
            att_state.ragged_mem_index,
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
        lengths = att_state.lengths // 4
        starts = att_state.query_batch * max_pools
        ends = starts + lengths
        groups = torch.empty((q.shape[0], self.topk // 4), dtype=torch.int32, device=q.device)
        # Bound the transient score matrix independently of total batch length.
        chunk_size = max(1, min(q.shape[0], 16 * 1024 * 1024 // max_pools))
        import deep_gemm

        pool_positions = torch.arange(max_pools, device=q.device)

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
            # The current image's fast_topk_v2 only supports 2048 entries;
            # K-pool selects 512 groups. Torch topk is CUDA-graph compatible.
            logits.masked_fill_(pool_positions[None, :] >= lengths[start:end, None], -float("inf"))
            groups[start:end] = torch.topk(logits, self.topk // 4, dim=-1, sorted=True).indices
        return expand_topk(groups, att_state.lengths, att_state.ks, att_state.ragged_mem_index, self.topk)
