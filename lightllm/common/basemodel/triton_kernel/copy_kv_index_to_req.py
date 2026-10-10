import torch
import triton
import triton.language as tl
from lightllm.common.basemodel.batch_objs import ModelInput


@triton.jit
def _build_kv_indexes_and_input_ids(
    Table,
    Req,
    Ends,
    Ready,
    Starts,
    Out,
    Tokens,
    Next,
    Mtp,
    Mixed,
    SB: tl.constexpr,
    SS: tl.constexpr,
    NEXT_STRIDE: tl.constexpr,
    BATCH: tl.constexpr,
    PREFILL: tl.constexpr,
    GATHER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pos = tl.arange(0, BLOCK)
    if PREFILL:
        batch = tl.program_id(0)
        req, end = tl.load(Req + batch), tl.load(Ends + batch)
        start, out_start = tl.load(Ready + batch), tl.load(Starts + batch)
        pos += tl.program_id(1) * BLOCK
        index = tl.load(Table + req * SB + (start + pos) * SS, pos < end - start, other=0)
        tl.store(Out + out_start + pos, index, pos < end - start)
        if GATHER:
            if tl.program_id(1) == 0 and tl.load(Mixed + batch):
                token = tl.load(Next + req * NEXT_STRIDE + tl.load(Mtp + batch))
                tl.store(Tokens + out_start, token)
    else:
        row = tl.program_id(0) * BLOCK + pos
        valid = row < BATCH
        req, end = tl.load(Req + row, valid, other=0), tl.load(Ends + row, valid, other=1)
        tl.store(Out + row, tl.load(Table + req * SB + (end - 1) * SS, valid, other=0), valid)
        if GATHER:
            step = tl.load(Mtp + row, valid, other=0)
            tl.store(Tokens + row, tl.load(Next + req * NEXT_STRIDE + step, valid, other=0), valid)


def build_kv_indexes_and_input_ids(model_input: ModelInput, req_to_token_indexs, next_token_ids=None):
    """Return KV write indexes and input IDs, gathering into supplied or newly allocated IDs in one launch."""
    prefill = model_input.is_prefill
    batch = model_input.batch_size
    b_req_idx = model_input.b_req_idx
    input_ids = model_input.input_ids
    max_q_seq_len = model_input.max_q_seq_len
    if input_ids is None and next_token_ids is not None:
        input_ids = torch.empty_like(b_req_idx, dtype=torch.int64)
    out = (
        torch.empty(input_ids.numel(), dtype=torch.int32, device=req_to_token_indexs.device)
        if prefill
        else torch.empty_like(b_req_idx, dtype=torch.int32)
    )
    block = min(triton.next_power_of_2(max(1, max_q_seq_len)), 1024) if prefill else 256
    grid = (batch, triton.cdiv(max_q_seq_len, block)) if prefill else (triton.cdiv(batch, block),)
    _build_kv_indexes_and_input_ids[grid](
        req_to_token_indexs,
        b_req_idx,
        model_input.b_seq_len,
        model_input.b_ready_cache_len if prefill else None,
        model_input.b_prefill_start_loc if prefill else None,
        out,
        input_ids,
        next_token_ids,
        model_input.b_mtp_index,
        model_input.b_is_decode_req,
        *req_to_token_indexs.stride(),
        0 if next_token_ids is None else next_token_ids.stride(0),
        batch,
        prefill,
        next_token_ids is not None,
        block,
        num_warps=max(1, block // 128) if prefill else 1,
    )
    return out, input_ids


@triton.jit
def _update_req_token_indexes(Table, Packet, STRIDE: tl.constexpr, BLOCK: tl.constexpr, PAGE: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    req = tl.load(Packet + 4 * row)
    start = tl.load(Packet + 4 * row + 1)
    size = tl.load(Packet + 4 * row + 2)
    offset = tl.load(Packet + 4 * row + 3)
    pos = block * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Packet + offset + pos // PAGE, pos < size, other=0) + pos % PAGE
    tl.store(Table + req * STRIDE + start + pos, value, pos < size)


def update_req_token_indexes(table, packet, count, max_size, page_size=1):
    _update_req_token_indexes[(count, triton.cdiv(max_size, 256))](
        table, packet, table.stride(0), 256, page_size, num_warps=4
    )
