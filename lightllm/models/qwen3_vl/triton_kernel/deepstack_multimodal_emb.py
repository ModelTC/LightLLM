import torch
import triton
import triton.language as tl
from lightllm.models.qwen3_vl.infer_struct import Qwen3VLInferStateInfo


@triton.jit
def _deepstack_add_kernel(
    input_ids,
    Deepstack_embs,
    Out,
    Img_token_lens,
    Img_start_token_ids,
    Img_start_locs_in_cache,
    stride_deep_s,
    stride_deep_d,
    stride_out_s,
    stride_out_d,
    hidden_size,
    BLOCK_DIM: tl.constexpr,
):
    seq_index = tl.program_id(0).to(tl.int64)
    img_handle_id = tl.program_id(1)

    token_id = tl.load(input_ids + seq_index)
    off_d = tl.arange(0, BLOCK_DIM)

    img_start_token_id = tl.load(Img_start_token_ids + img_handle_id)
    img_token_len = tl.load(Img_token_lens + img_handle_id)
    img_start_loc_in_cache = tl.load(Img_start_locs_in_cache + img_handle_id)

    # 判断当前 token 是否属于这个 image
    cond = (token_id >= img_start_token_id) & (token_id < img_start_token_id + img_token_len)

    for _ in range(0, tl.where(cond, 1, 0), 1):
        token_offset = token_id - img_start_token_id

        deep_row = tl.load(
            Deepstack_embs + stride_deep_s * (img_start_loc_in_cache + token_offset) + off_d,
            mask=off_d < hidden_size,
            other=0,
        )
        old = tl.load(
            Out + stride_out_s * seq_index + off_d,
            mask=off_d < hidden_size,
            other=0,
        )
        tl.store(
            Out + stride_out_s * seq_index + off_d,
            old + deep_row,
            mask=off_d < hidden_size,
        )
    return


@torch.no_grad()
def add_deepstack_embs(
    out: torch.Tensor,
    input_ids: torch.Tensor,
    deepstack_embs: torch.Tensor,
    img_token_lens: torch.Tensor,
    img_start_token_ids: torch.Tensor,
    img_start_locs_in_cache: torch.Tensor,
):
    assert input_ids.dim() == 1
    assert out.dim() == 2
    assert deepstack_embs.dim() == 2

    total_len = input_ids.shape[0]
    hidden = out.shape[1]
    BLOCK = triton.next_power_of_2(hidden)

    grid = (total_len, img_token_lens.shape[0])
    num_warps = 4

    _deepstack_add_kernel[grid](
        input_ids,
        deepstack_embs,
        out,
        img_token_lens,
        img_start_token_ids,
        img_start_locs_in_cache,
        deepstack_embs.stride(0),
        deepstack_embs.stride(1),
        out.stride(0),
        out.stride(1),
        hidden_size=hidden,
        BLOCK_DIM=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return


@torch.no_grad()
def apply_deepstack_features(
    input_embeddings: torch.Tensor,
    infer_state: Qwen3VLInferStateInfo,
    layer_num: int,
):
    """
    apply deepstack features for all images in qwen3-vl/qwen3-vl-moe
    """

    if not infer_state.deepstack_features:
        return

    deepstack_num_layers = infer_state.cpu_embed_cache_tensor.shape[1] - 1

    if layer_num >= deepstack_num_layers:
        return

    add_deepstack_embs(
        out=input_embeddings,
        input_ids=infer_state.input_ids,
        deepstack_embs=infer_state.cpu_embed_cache_tensor[:, layer_num + 1, :],
        img_token_lens=infer_state.img_token_lens,
        img_start_token_ids=infer_state.img_start_token_ids,
        img_start_locs_in_cache=infer_state.img_start_locs_in_cache,
    )
    return
