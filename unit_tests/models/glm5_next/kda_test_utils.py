"""Build KDA adapter inputs the way GLM-5-Next layers do (see _kda_att_dict)."""

from types import SimpleNamespace

import torch

from lightllm.common.attention.backend import AttentionBatch


def kda_params(conv_states, ssm_states, layer_weight, mixed_qkv, raw_gate, raw_beta, *, state=None, layer_num=0):
    return dict(
        mixed_qkv=mixed_qkv,
        raw_gate=raw_gate,
        raw_beta=raw_beta,
        conv_weight=layer_weight.get_merged_kda_conv_weight(),
        A_log=layer_weight.linear_A_log.weight,
        dt_bias=layer_weight.linear_dt_bias.weight,
        conv_states=conv_states,
        ssm_states=ssm_states,
        state=state,
        layer_num=layer_num,
    )


def prepare_prefill_infer(infer, device="cuda"):
    """Fill the request fields a prefill step builder reads but a test stub does not set."""
    infer.is_prefill = True
    for name in ("is_cuda_graph_capture", "is_cuda_graph_replay"):
        setattr(infer, name, getattr(infer, name, False))
    infer.batch_size = infer.b_req_idx.shape[0]
    infer.b_seq_len = getattr(infer, "b_seq_len", None)
    infer.max_kv_seq_len = getattr(infer, "max_kv_seq_len", None)
    infer.total_token_num = getattr(infer, "total_token_num", None)
    infer.b1_cu_kv_seq_len = getattr(infer, "b1_cu_kv_seq_len", None)
    infer.b_q_start_loc = getattr(infer, "b_q_start_loc", None)
    infer.b_ready_cache_len = getattr(infer, "b_ready_cache_len", None)
    infer.max_cache_len = getattr(infer, "max_cache_len", 0)
    infer.mem_index = getattr(infer, "mem_index", None)
    infer.microbatch_index = getattr(infer, "microbatch_index", 0)
    infer.req_manager = getattr(infer, "req_manager", SimpleNamespace(req_to_token_indexs=None))
    return infer


def prefill_batch(max_q_seq_len, device="cpu"):
    b_req_idx = torch.zeros(1, dtype=torch.int32, device=device)
    return AttentionBatch(
        req_to_token_indexs=None,
        b_req_idx=b_req_idx,
        b_seq_len=b_req_idx,
        batch_size=1,
        max_kv_seq_len=max_q_seq_len,
        total_token_num=max_q_seq_len,
        cu_q_seq_len=torch.tensor([0, max_q_seq_len], dtype=torch.int32, device=device),
        b_ready_cache_len=b_req_idx,
        max_q_seq_len=max_q_seq_len,
    )
