"""Prefill scratch for GLM-5 sparse attention over the packed FP8 cache.

FA3 sparse MLA prefill dequantizes the selected keys chunk by chunk into a fixed BF16 workspace.
The model reserves it, one region per prefill microbatch, before it sizes the KV cache, and hands it
to sparse prefill layers through their model.
"""

import torch
import triton

from lightllm.utils.workspace_utils import get_gpu_workspace_buffer

# Cap each microbatch's BF16 KV workspace at 2 GiB (2097152 × 512 × 2 bytes).
KV_GATHER_CHUNK_SIZE = 2097152


def prepare_sparse_prefill_workspaces(model) -> None:
    if model.args.llm_kv_type == "fp8kv_dsa" and model.att_state_manager.get_full_attention_states(
        model.att_state_role
    ):
        regions = reserve_fp8_sparse_prefill_workspaces(model)
        for layer in model.layers_infer:
            if not layer.is_linear_attention_layer:
                layer.fp8_sparse_prefill_workspaces = regions


def reserve_fp8_sparse_prefill_workspaces(model) -> list[dict[str, torch.Tensor]]:
    """Reserve one workspace region per prefill microbatch for the FA3 sparse MLA kernel.

    Regions are named by microbatch, not by model role: the main and draft models prefill one after the
    other and microbatch i of every model runs on the same stream, so a draft model reuses the main
    model's reservation (a prefix view when it needs less) instead of reserving its own.
    """
    regions = 1 + int(model.args.enable_prefill_microbatch_overlap)
    q_token_capacity = model.batch_max_tokens
    topk_size = triton.cdiv(model.config["index_topk"] + 3, 128) * 128
    kv_token_capacity = min(KV_GATHER_CHUNK_SIZE, model.max_req_num * model.max_seq_length + q_token_capacity)
    if model.args.run_mode == "decode":
        kv_token_capacity = min(kv_token_capacity, q_token_capacity)
    # Query offsets of one-token requests; read-only, so every region and model reads a prefix of one arange.
    cu_seqlens_q = get_gpu_workspace_buffer("glm5_fp8_prefill_cu_q", q_token_capacity + 1, torch.int32)
    cu_seqlens_q.copy_(torch.arange(q_token_capacity + 1, dtype=torch.int32, device=cu_seqlens_q.device))
    workspaces = []
    for region in range(regions):
        name = f"glm5_fp8_prefill_{region}"
        gathered_kv = get_gpu_workspace_buffer(f"{name}_kv", kv_token_capacity * 512, torch.bfloat16)
        chunk_topk_indices = get_gpu_workspace_buffer(f"{name}_topk", q_token_capacity * topk_size, torch.int32)
        workspaces.append(
            {
                "gathered_kv": gathered_kv.view(kv_token_capacity, 1, 512),
                "chunk_topk_indices": chunk_topk_indices.view(q_token_capacity, topk_size),
                "chunk_cache_seqlens": get_gpu_workspace_buffer(f"{name}_seqlens", q_token_capacity, torch.int32),
                "cu_seqlens_q": cu_seqlens_q,
            }
        )
    return workspaces
