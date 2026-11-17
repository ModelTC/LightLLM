import torch

from lightllm.common.attention.backend import AttentionBackend, TensorAllocator
from . import DenseInputs, DensePrefillBatch, DenseDecodeBatch


class TritonDensePrefillBackend(AttentionBackend[DensePrefillBatch, DenseInputs]):
    def forward(
        self,
        batch: DensePrefillBatch,
        inputs: DenseInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out, with q's shape, dtype and device.

        out must have unit head-dimension stride and non-overlapping token/head
        strides; padding is allowed. It must not alias inputs or batch tensors.
        Head dimensions 16, 32, 64, 128, 256 and 512 are supported.
        Prefill needs no scratch and does not call alloc_func.
        """
        from lightllm.common.attention.kernels.dense.prefill.context_flashattention_nopad import context_attention_fwd

        context_attention_fwd(
            q=inputs.q,
            k=inputs.k_cache,
            v=inputs.v_cache,
            o=out,
            b_req_idx=batch.b_req_idx,
            b_start_loc=batch.b_q_start_loc,
            b_seq_len=batch.b_seq_len,
            b_prompt_cache_len=batch.b_ready_cache_len,
            max_input_len=batch.max_q_seq_len,
            req_to_token_indexs=batch.req_to_token_indexs,
            sliding_window=inputs.sliding_window,
        )
        return out


class TritonDenseDecodeBackend(AttentionBackend[DenseDecodeBatch, DenseInputs]):
    def forward(
        self,
        batch: DenseDecodeBatch,
        inputs: DenseInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out, with q's shape, dtype and device.

        out must have unit head-dimension stride and non-overlapping token/head
        strides; padding is allowed. It must not alias inputs or batch tensors.
        Head dimensions 16, 32, 64, 128 and 256 are supported, plus 512 for GQA.
        Scratch uses alloc_func on q.device. A serving allocator with a cache
        scoped to one device must be used only on that device.
        """
        q_head_num = inputs.q.shape[1]
        k_head_num = inputs.k_cache.shape[1]
        if q_head_num == k_head_num:
            assert inputs.sliding_window == (-1, -1), "sliding_window not supported in non-gqa attention yet"
            from lightllm.common.attention.kernels.dense.decode.mha.flash_decoding import (
                token_decode_attention_flash_decoding,
            )

            return token_decode_attention_flash_decoding(
                q=inputs.q,
                cache_k=inputs.k_cache,
                cache_v=inputs.v_cache,
                req_to_token_indexs=batch.req_to_token_indexs,
                b_req_idx=batch.b_req_idx,
                b_seq_len=batch.b_seq_len,
                max_kv_seq_len=batch.kv_seq_len_upper_bound,
                out=out,
                alloc_tensor_func=alloc_func,
            )
        elif q_head_num > k_head_num:
            from lightllm.common.attention.kernels.dense.decode.gqa.gqa_flash_decoding import (
                gqa_token_decode_attention_flash_decoding,
            )

            return gqa_token_decode_attention_flash_decoding(
                q=inputs.q,
                cache_k=inputs.k_cache,
                cache_v=inputs.v_cache,
                req_to_token_indexs=batch.req_to_token_indexs,
                b_req_idx=batch.b_req_idx,
                b_seq_len=batch.b_seq_len,
                max_len_in_batch=batch.tuning_kv_seq_len,
                out=out,
                alloc_tensor_func=alloc_func,
                sliding_window=inputs.sliding_window,
            )
        else:
            raise NotImplementedError("error")
