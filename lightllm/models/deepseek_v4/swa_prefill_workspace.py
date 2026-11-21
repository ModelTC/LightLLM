"""Sliding-window cache staging for DeepSeek V4 prefill.

A prefill writes each request's window into a workspace instead of the runtime SWA cache, so long
prompts can attend to keys the ring would already have overwritten. The model reserves the
workspace before sizing the KV cache.
"""

from dataclasses import dataclass

import torch
import triton

from lightllm.common.basemodel.attention.base_att import BaseAttBackend


@dataclass
class SwaPrefillWorkspace:
    """One prefill step's SWA workspace.

    swa_buffer is [layers, slots, packed_bytes]; HOLD_SWA_SLOT is its trailing page for the HOLD
    request. Per request slot, prefill_token_ranges holds [ready, end) positions written to the
    workspace and swa_prefill_workspace_offsets the request's first workspace slot (-1 unused).
    """

    swa_buffer: torch.Tensor
    HOLD_SWA_SLOT: int
    prefill_token_ranges: torch.Tensor
    swa_prefill_workspace_offsets: torch.Tensor


def reserve_swa_prefill_workspace(model):
    """Return the (possibly shared) workspace buffer and the token capacity of each microbatch region."""
    from lightllm.models.deepseek_v4.config import create_attention_config

    config = create_attention_config(
        model_config=model.config,
        model_logical_layer_ids=tuple(layer.layer_num_ for layer in model.layers_infer),
        tp_rank=model.tp_rank_,
        tp_world_size=model.tp_world_size_,
    ).create_swa_config()
    page_token_num = config.page_token_size
    max_forward_tokens = max(model.batch_max_tokens, model.max_req_num) + 2 * model.tp_world_size_
    capacity = (
        triton.cdiv(max_forward_tokens + model.max_req_num * (config.window_size + 2 * page_token_num), page_token_num)
        * page_token_num
    )
    if model.args.run_mode == "decode":
        # D 节点初始化仍会对 HOLD 请求执行一次 prefill；只保留末尾的 HOLD 页，
        # 不为普通请求预留长 prefill 临时区，实际请求的前缀由 P 节点传入。
        capacity = 0
    regions = 1 + int(model.args.enable_prefill_microbatch_overlap)
    # 每个 microbatch 使用独立临时区；最后一页留给 HOLD 请求。
    token_num = regions * capacity + page_token_num
    # 物理行宽由 SWA 原子配置给出，这里仅决定 workspace 的 token 容量。
    slot_bytes = config.get_packed_kv_bytes(page_token_num) // page_token_num
    layer_count = len(config.logical_layer_ids)
    buffer = BaseAttBackend.get_gpu_workspace_buffer(
        key_name=f"deepseek_v4_swa_prefill_{model.att_state_role}",
        workspace_size=layer_count * token_num * slot_bytes,
        dtype=torch.uint8,
    )
    return buffer.view(layer_count, token_num, slot_bytes), capacity


def prepare_swa_prefill_workspace(infer_state, model) -> SwaPrefillWorkspace:
    """Stage the step's requests into the workspace region of the step's microbatch."""
    from lightllm.models.deepseek_v4.triton_kernel.swa_cache import prepare_swa_prefill_workspace as stage

    swa_state = infer_state.swa_state
    swa_buffer, capacity = reserve_swa_prefill_workspace(model)
    workspace = SwaPrefillWorkspace(
        swa_buffer=swa_buffer,
        HOLD_SWA_SLOT=swa_buffer.shape[1] - swa_state.config.page_token_size,
        prefill_token_ranges=torch.zeros((swa_state.request_capacity, 2), dtype=torch.int32, device=swa_buffer.device),
        swa_prefill_workspace_offsets=torch.full(
            (swa_state.request_capacity,), -1, dtype=torch.int32, device=swa_buffer.device
        ),
    )
    stage(
        swa_state=swa_state,
        prefill_state=workspace,
        b_req_idx=infer_state.b_req_idx,
        b_ready_cache_len=infer_state.b_ready_cache_len,
        b_seq_len=infer_state.b_seq_len,
        workspace_base=infer_state.microbatch_index * capacity,
    )
    return workspace
