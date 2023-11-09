import torch

def token_decode_attention_flash_decoding(q, infer_state, q_head_num, head_dim, cache_k, cache_v):
    BLOCK_SEQ = 256
    batch_size = infer_state.batch_size
    max_len_in_batch = infer_state.max_len_in_batch
    calcu_shape1 = (batch_size, q_head_num, head_dim)

    from .flash_decoding_stage1 import flash_decode_stage1
    from .flash_decoding_stage2 import flash_decode_stage2

    o_tensor = torch.empty_like(q)
    
    if getattr(infer_state, 'mid_o', None) is None:
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

    flash_decode_stage1(q.view(calcu_shape1),
                                cache_k,
                                cache_v,
                                infer_state.req_manager.req_to_token_indexs,
                                infer_state.b_req_idx,
                                infer_state.b_seq_len,
                                infer_state.max_len_in_batch,
                                mid_o,
                                mid_o_logexpsum,
                                BLOCK_SEQ)
    flash_decode_stage2(mid_o,
                        mid_o_logexpsum, 
                        infer_state.b_seq_len, 
                        o_tensor.view(calcu_shape1), 
                        BLOCK_SEQ)
    return o_tensor
