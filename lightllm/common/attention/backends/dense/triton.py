import torch

from lightllm.common.attention.backend import AttentionBackend
from . import DenseInputs, DensePrefillBatch, DenseDecodeBatch


class TritonDensePrefillBackend(AttentionBackend[DensePrefillBatch, DenseInputs]):
    def forward(self, batch, inputs, *, out, alloc_func=torch.empty):
        from lightllm.common.attention.kernels.dense.prefill.context_flashattention_nopad import context_attention_fwd

        context_attention_fwd(
            inputs.q,
            inputs.k,
            inputs.v,
            out,
            batch.b_req_idx,
            batch.b_q_start_loc,
            batch.b_seq_len,
            batch.b_ready_cache_len,
            batch.max_q_seq_len,
            batch.req_to_token_indexs,
            sliding_window=inputs.sliding_window,
        )
        return out


class TritonDenseDecodeBackend(AttentionBackend[DenseDecodeBatch, DenseInputs]):
    def forward(self, batch, inputs, *, out, alloc_func=torch.empty):
        q_head_num = inputs.q.shape[1]
        k_head_num = inputs.k.shape[1]
        if q_head_num == k_head_num:
            assert inputs.sliding_window == (-1, -1), "sliding_window not supported in non-gqa attention yet"
            from lightllm.common.attention.kernels.dense.decode.mha.flash_decoding import (
                token_decode_attention_flash_decoding,
            )

            return token_decode_attention_flash_decoding(
                q=inputs.q,
                cache_k=inputs.k,
                cache_v=inputs.v,
                req_to_token_indexs=batch.req_to_token_indexs,
                b_req_idx=batch.b_req_idx,
                b_seq_len=batch.b_seq_len,
                max_kv_seq_len=batch.max_kv_seq_len,
                out=out,
                alloc_tensor_func=alloc_func,
            )
        elif q_head_num > k_head_num:
            from lightllm.common.attention.kernels.dense.decode.gqa.gqa_flash_decoding import (
                gqa_token_decode_attention_flash_decoding,
            )

            return gqa_token_decode_attention_flash_decoding(
                q=inputs.q,
                cache_k=inputs.k,
                cache_v=inputs.v,
                req_to_token_indexs=batch.req_to_token_indexs,
                b_req_idx=batch.b_req_idx,
                b_seq_len=batch.b_seq_len,
                max_len_in_batch=batch.max_len_in_batch,
                out=out,
                alloc_tensor_func=alloc_func,
                sliding_window=inputs.sliding_window,
            )
        else:
            raise NotImplementedError("error")
