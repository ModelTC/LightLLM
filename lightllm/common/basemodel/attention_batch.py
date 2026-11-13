from typing import TYPE_CHECKING

from lightllm.common.attention.api import AttentionBatch, AttentionPhase

if TYPE_CHECKING:
    from .infer_struct import InferStateInfo


class AttentionBatchBuilder:
    @staticmethod
    def from_infer_state(state: "InferStateInfo") -> AttentionBatch:
        """Borrow input tensors; capture retains their addresses and replay copies only their contents."""
        return AttentionBatch(
            phase=AttentionPhase.PREFILL if state.is_prefill else AttentionPhase.DECODE,
            batch_size=state.batch_size,
            token_num=state.input_ids.shape[0],
            total_token_num=state.total_token_num,
            max_q_seq_len=state.max_q_seq_len,
            max_kv_seq_len=state.max_kv_seq_len,
            max_cache_len=state.max_cache_len or 0,
            b_req_idx=state.b_req_idx,
            b_seq_len=state.b_seq_len,
            b_q_seq_len=state.b_q_seq_len,
            b1_cu_q_seq_len=state.b1_cu_q_seq_len,
            b1_cu_kv_seq_len=state.b1_cu_kv_seq_len,
            req_to_token_indexs=state.req_manager.req_to_token_indexs,
            hold_req_id=state.req_manager.HOLD_REQUEST_ID,
            b_ready_cache_len=state.b_ready_cache_len,
            b_mtp_index=state.b_mtp_index,
            b_shared_seq_len=state.b_shared_seq_len,
            b_shared_radix_node_id=state.b_shared_radix_node_id,
            mem_index=state.mem_index,
            position_ids=state.position_ids,
            b_image_token_end=getattr(state, "b_image_token_end", None),
        )
