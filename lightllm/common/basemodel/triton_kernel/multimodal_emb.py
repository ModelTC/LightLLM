import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Prompt_ids,
    Text_weight_embs,
    Embed_cache,
    Out,
    Img_token_lens,
    Img_start_token_ids,
    Img_start_locs_in_cache,
    stride_text_emb_s,
    stride_text_emb_d,  # text_stride
    stride_emb_cache_s,
    stride_emb_cache_l,
    stride_emb_cache_d,  # img_stride
    stride_out_s,
    stride_out_d,
    tp_text_start_token_id,
    tp_text_end_token_id,
    hidden_size,
    tp_world_size,
    BLOCK_HIDDEN_DIM: tl.constexpr,
):

    seq_index = tl.program_id(0).to(tl.int64)
    img_handle_id = tl.program_id(1)

    token_id = tl.load(Prompt_ids + seq_index)
    off_d = tl.arange(0, BLOCK_HIDDEN_DIM)

    # load store text emb
    for _ in range(
        0,
        tl.where((img_handle_id == 0) & (token_id < tp_text_end_token_id) & (token_id >= tp_text_start_token_id), 1, 0),
        1,
    ):
        load_emb = tl.load(
            Text_weight_embs + stride_text_emb_s * (token_id - tp_text_start_token_id) + off_d * stride_text_emb_d,
            mask=off_d < hidden_size,
            other=0,
        )
        tl.store(Out + stride_out_s * seq_index + stride_out_d * off_d, load_emb, mask=off_d < hidden_size)

    img_start_token_id = tl.load(Img_start_token_ids + img_handle_id - 1, mask=img_handle_id >= 1, other=0)
    img_start_loc = tl.load(Img_start_locs_in_cache + img_handle_id - 1, mask=img_handle_id >= 1, other=0)
    img_token_len = tl.load(Img_token_lens + img_handle_id - 1, mask=img_handle_id >= 1, other=0)
    # load store img emb
    for _ in range(
        0,
        tl.where(
            (img_handle_id != 0) & (token_id >= img_start_token_id) & (token_id < img_start_token_id + img_token_len),
            1,
            0,
        ),
        1,
    ):
        load_emb = tl.load(
            Embed_cache
            + stride_emb_cache_s.to(tl.int64) * (img_start_loc + token_id - img_start_token_id)
            + stride_emb_cache_l * 0
            + stride_emb_cache_d * off_d,
            mask=off_d < hidden_size,
            other=0,
        )
        tl.store(
            Out + stride_out_s * seq_index + stride_out_d * off_d, load_emb / tp_world_size, mask=off_d < hidden_size
        )
    return


@torch.no_grad()
def multimodal_emb(
    out: torch.Tensor,
    prompt_ids: torch.Tensor,
    text_weight_embs: torch.Tensor,
    embed_cache: torch.Tensor,
    img_token_lens: torch.Tensor,
    img_start_token_ids: torch.Tensor,
    img_start_locs_in_cache: torch.Tensor,
    tp_text_start_token_id: int,
    tp_text_end_token_id: int,
    tp_world_size: int,
):
    total_len = prompt_ids.shape[0]
    BLOCK = triton.next_power_of_2(out.shape[1])
    # print(len(img_token_lens))
    grid = (total_len, len(img_token_lens) + 1)
    num_warps = 1
    _fwd_kernel[grid](
        Prompt_ids=prompt_ids,
        Text_weight_embs=text_weight_embs,
        Embed_cache=embed_cache,
        Out=out,
        Img_token_lens=img_token_lens,
        Img_start_token_ids=img_start_token_ids,
        Img_start_locs_in_cache=img_start_locs_in_cache,
        stride_text_emb_s=text_weight_embs.stride(0),
        stride_text_emb_d=text_weight_embs.stride(1),
        stride_emb_cache_s=embed_cache.stride(0),
        stride_emb_cache_l=embed_cache.stride(1),
        stride_emb_cache_d=embed_cache.stride(2),
        stride_out_s=out.stride(0),
        stride_out_d=out.stride(1),
        tp_text_start_token_id=tp_text_start_token_id,
        tp_text_end_token_id=tp_text_end_token_id,
        hidden_size=out.shape[1],
        tp_world_size=float(tp_world_size),
        BLOCK_HIDDEN_DIM=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _mark_multimodal_obj_need_kernel(
    obj_start_token_ids_ptr,
    obj_token_lens_ptr,
    obj_marks_ptr,
    input_ids_ptr,
    input_size,
    BLOCK_SIZE: tl.constexpr,
):

    obj_index = tl.program_id(0)
    start_id = tl.load(obj_start_token_ids_ptr + obj_index)
    token_len = tl.load(obj_token_lens_ptr + obj_index)

    for block_start in range(0, input_size, BLOCK_SIZE):
        block_range = block_start + tl.arange(0, BLOCK_SIZE)
        cur_input_ids = tl.load(input_ids_ptr + block_range, mask=block_range < input_size, other=0)
        mark = tl.where((cur_input_ids >= start_id) & (cur_input_ids < start_id + token_len), 1, 0)
        mark = tl.sum(mark)
        tl.store(obj_marks_ptr + obj_index, 1, mask=mark > 0)
    return


@torch.no_grad()
def mark_multimodal_obj(obj_start_token_ids: torch.Tensor, obj_token_lens: torch.Tensor, input_ids: torch.Tensor):
    out_mark = torch.empty_like(obj_start_token_ids)
    out_mark.fill_(0)
    assert obj_start_token_ids.shape == obj_token_lens.shape
    BLOCK = 512
    grid = (obj_start_token_ids.shape[0],)
    _mark_multimodal_obj_need_kernel[grid](
        obj_start_token_ids_ptr=obj_start_token_ids,
        obj_token_lens_ptr=obj_token_lens,
        obj_marks_ptr=out_mark,
        input_ids_ptr=input_ids,
        input_size=input_ids.shape[0],
        BLOCK_SIZE=BLOCK,
        num_warps=1,
        num_stages=1,
    )
    return out_mark
