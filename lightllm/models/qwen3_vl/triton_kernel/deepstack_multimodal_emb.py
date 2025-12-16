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
    Img_start_locs,
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

    img_start_token_id = tl.load(
        Img_start_token_ids + img_handle_id,
        mask=img_handle_id >= 0,
        other=0,
    )
    img_start_loc = tl.load(
        Img_start_locs + img_handle_id,
        mask=img_handle_id >= 0,
        other=0,
    )
    img_token_len = tl.load(
        Img_token_lens + img_handle_id,
        mask=img_handle_id >= 0,
        other=0,
    )

    # 判断当前 token 是否属于这个 image
    cond = (token_id >= img_start_token_id) & (token_id < img_start_token_id + img_token_len)

    for _ in range(0, tl.where(cond, 1, 0), 1):
        token_offset = token_id - img_start_token_id

        # 从 Deepstack_embs 里取对应行
        deep_row = tl.load(
            Deepstack_embs + stride_deep_s * (img_start_loc + token_offset) + off_d * stride_deep_d,
            mask=off_d < hidden_size,
            other=0,
        )

        # 把 deepstack 加到 Out[seq_index] 上
        old = tl.load(
            Out + stride_out_s * seq_index + stride_out_d * off_d,
            mask=off_d < hidden_size,
            other=0,
        )
        tl.store(
            Out + stride_out_s * seq_index + stride_out_d * off_d,
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
    img_start_locs: torch.Tensor,
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
        img_start_locs,
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

    deepstack_layers_num = len(infer_state.deepstack_features[0])

    if layer_num < 0 or layer_num >= deepstack_layers_num:
        return

    input_ids = infer_state.input_ids
    device = input_embeddings.device

    if infer_state.img_token_lens.shape[0] == 0:
        return

    per_img_deepstack_features = [
        infer_state.deepstack_features[i][layer_num].to(device=device, non_blocking=True)
        for i in range(infer_state.img_token_lens.shape[0])
    ]
    all_deepstack_features = torch.cat(per_img_deepstack_features, dim=0)
    add_deepstack_embs(
        out=input_embeddings,
        input_ids=input_ids,
        deepstack_embs=all_deepstack_features,
        img_token_lens=infer_state.img_token_lens,
        img_start_token_ids=infer_state.img_start_token_ids,
        img_start_locs=infer_state.img_start_locs,
    )
    return
