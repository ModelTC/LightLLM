import time
import torch
import numpy as np
from lightllm.common.basemodel import InferStateInfo

def gqa_token_decode_attention_flash_decoding(q, infer_state:InferStateInfo, q_head_num, head_dim, cache_k, cache_v, out=None):
    BLOCK_SEQ = 128
    batch_size = infer_state.batch_size
    max_len_in_batch = infer_state.max_len_in_batch
    calcu_shape1 = (batch_size, q_head_num, head_dim)

    from .gqa_flash_decoding_stage1 import flash_decode_stage1
    from .gqa_flash_decoding_stage2 import flash_decode_stage2

    o_tensor = torch.empty_like(q) if out is None else out
    
    if getattr(infer_state, 'mid_o', None) is None:
        # start_time = time.time()
        b_seq_len_numpy = infer_state.b_seq_len.cpu().numpy()

        block_batch_ids = torch.from_numpy(np.concatenate([np.full(((b_seq_len_numpy[batch_id] + BLOCK_SEQ - 1) // BLOCK_SEQ,), fill_value=batch_id, dtype=np.int32) 
                                         for batch_id in range(len(b_seq_len_numpy))], axis=0)).cuda()
        
        block_start_indexes = torch.from_numpy(np.concatenate([np.arange(0, seq_len, BLOCK_SEQ, dtype=np.int32)
                                         for seq_len in b_seq_len_numpy], axis=0)).cuda()
        
        assert len(block_batch_ids) == len(block_start_indexes)
        infer_state.block_batch_ids = block_batch_ids
        infer_state.block_start_indexes = block_start_indexes
        # print("build block params cost:", (time.time() - start_time) * 1000)

        infer_state.mid_o = torch.empty([batch_size, 
                                        q_head_num, 
                                        max_len_in_batch // BLOCK_SEQ + 1, 
                                        head_dim], 
                                        dtype=torch.float32, 
                                        device="cuda")
        infer_state.mid_o_logexpsum = torch.empty([batch_size, 
                                        q_head_num,
                                        max_len_in_batch // BLOCK_SEQ + 1], 
                                        dtype=torch.float32, 
                                        device="cuda")
        
    mid_o = infer_state.mid_o
    mid_o_logexpsum = infer_state.mid_o_logexpsum

    flash_decode_stage1(infer_state.block_batch_ids,
                        infer_state.block_start_indexes,
                        q.view(calcu_shape1),
                        cache_k,
                        cache_v,
                        infer_state.req_manager.req_to_token_indexs,
                        infer_state.b_req_idx,
                        infer_state.b_seq_len,
                        mid_o,
                        mid_o_logexpsum,
                        BLOCK_SEQ)
    flash_decode_stage2(mid_o,
                        mid_o_logexpsum, 
                        infer_state.b_seq_len, 
                        o_tensor.view(calcu_shape1), 
                        BLOCK_SEQ)
    return o_tensor
