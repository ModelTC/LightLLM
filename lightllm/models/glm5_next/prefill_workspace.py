"""Prefill scratch for GLM-5 sparse attention over the packed FP8 cache.

FA3 sparse MLA prefill dequantizes the selected keys chunk by chunk into a fixed BF16 workspace.
The model reserves it, one region per prefill microbatch, before it sizes the KV cache, and hands it
to sparse prefill layers through their model.
"""

import torch
import triton

from lightllm.common.attention.base_backend import PrefillAttBackend

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
    """Reserve one workspace region per prefill microbatch for the FA3 sparse MLA kernel."""
    regions = 1 + int(model.args.enable_prefill_microbatch_overlap)
    q_token_capacity = model.batch_max_tokens
    topk_size = triton.cdiv(x=model.config["index_topk"] + 3, y=128) * 128
    kv_token_capacity = min(KV_GATHER_CHUNK_SIZE, model.max_req_num * model.max_seq_length + q_token_capacity)
    if model.args.run_mode == "decode":
        kv_token_capacity = min(kv_token_capacity, q_token_capacity)
    role = model.att_state_role
    get_workspace = PrefillAttBackend.get_workspace
    gathered_kv = get_workspace(
        name="glm5_fp8_prefill_kv", role=role, numel=regions * kv_token_capacity * 512, dtype=torch.bfloat16
    )
    gathered_kv = gathered_kv.view(regions, kv_token_capacity, 1, 512)
    chunk_topk_indices = get_workspace(
        name="glm5_fp8_prefill_topk", role=role, numel=regions * q_token_capacity * topk_size, dtype=torch.int32
    ).view(regions, q_token_capacity, topk_size)
    chunk_cache_seqlens = get_workspace(
        name="glm5_fp8_prefill_seqlens", role=role, numel=regions * q_token_capacity, dtype=torch.int32
    ).view(regions, q_token_capacity)
    # Q offsets are written once per role. Keep them separate so roles with different capacities cannot overwrite them.
    cu_seqlens_q = get_workspace(
        name=f"glm5_fp8_prefill_cu_q_{role}", role=role, numel=regions * (q_token_capacity + 1), dtype=torch.int32
    ).view(regions, q_token_capacity + 1)
    cu_seqlens_q.copy_(torch.arange(q_token_capacity + 1, dtype=torch.int32, device=cu_seqlens_q.device))
    return [
        {
            "gathered_kv": gathered_kv[index],
            "chunk_topk_indices": chunk_topk_indices[index],
            "chunk_cache_seqlens": chunk_cache_seqlens[index],
            "cu_seqlens_q": cu_seqlens_q[index],
        }
        for index in range(regions)
    ]
