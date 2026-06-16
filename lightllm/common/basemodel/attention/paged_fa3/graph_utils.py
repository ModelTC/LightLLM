import torch
from typing import Any

SEQ_LEN_REF_UNINITIALIZED = -1


def new_seq_len_ref_buffers(max_batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.full((max_batch_size,), SEQ_LEN_REF_UNINITIALIZED, dtype=torch.int32),
        torch.full((max_batch_size,), SEQ_LEN_REF_UNINITIALIZED, dtype=torch.int32),
    )


def maybe_sync_attn_params(
    *,
    batch_size: int,
    b1_cu_q_seq_len_cpu_slice: torch.Tensor,
    b_cu_kv_seq_len_cpu_slice: torch.Tensor,
    b1_cu_q_seq_len_cpu: torch.Tensor,
    b_cu_kv_seq_len_cpu: torch.Tensor,
    update_stream: Any,
) -> None:
    if batch_size == 0:
        return
    if torch.equal(b_cu_kv_seq_len_cpu_slice, b_cu_kv_seq_len_cpu):
        return

    b1_cu_q_seq_len_cpu_slice.copy_(b1_cu_q_seq_len_cpu)
    b_cu_kv_seq_len_cpu_slice.copy_(b_cu_kv_seq_len_cpu)
    update_attn_params(
        batch_size,
        b1_cu_q_seq_len_cpu_slice,
        b_cu_kv_seq_len_cpu_slice,
        update_stream,
    )


def weak_ref_tensor(tensor: Any) -> Any:
    import torch_npu

    if isinstance(tensor, torch.Tensor):
        return torch_npu._C._weak_ref_tensor(tensor)
    return tensor


def update_attn_params(
    batch_size: int,
    actual_seq_lengths: list[int],
    actual_seq_lengths_kv: list[int],
    update_stream: Any,
):
    import torch_npu
    from lightllm.common.basemodel.graph.acl_graph import get_attn_params

    attn_params = get_attn_params()
    handles = attn_params.handles[batch_size]
    events = attn_params.events[batch_size]
    workspace = attn_params.workspaces[batch_size]
    params_list = attn_params.attn_params[batch_size]

    with torch.npu.stream(update_stream):
        for handle, event, attn_param in zip(handles, events, params_list):
            (q, k, v, sm_scale, N_Q, N_KV, page_table, block_size, output, softmax_lse) = attn_param
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu.npu_fused_infer_attention_score.out(
                q,
                k,
                v,
                input_layout="TND",
                scale=sm_scale,
                actual_seq_lengths=actual_seq_lengths,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_heads=N_Q,
                num_key_value_heads=N_KV,
                block_table=page_table,
                block_size=block_size,
                workspace=workspace,
                out=[output, softmax_lse],
            )
            torch.npu.graph_task_update_end(update_stream)
            event.record(update_stream)
