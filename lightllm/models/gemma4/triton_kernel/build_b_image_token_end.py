"""GPU-resident builder for ``b_image_token_end``.

Replaces a 3× D2H sync + Python per-batch-image slice-fill in CPU memory
with a single small H2D copy (image metadata) + one Triton kernel that
scatters the image-end markers into the flat-Q-token tensor on GPU.

Adapted from neo_chat_moe's `get_neo_position_triton`. Same per-batch
program structure; we only emit the `b_image_token_end` scatter (no 3D
position_ids — gemma-4 uses 1D position ids).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _build_b_image_token_end_kernel(
    B_Image_Start_Idx,  # (num_imgs,) int32, image span start in absolute request position
    B_Image_Len,  # (num_imgs,) int32, image token count
    B_Image_Nums,  # (batch,)    int32, per-batch image count
    B_Image_Start_Num,  # (batch,)    int32, prefix-sum offset into flat per-image arrays
    B_Q_Start_Loc,  # (batch,)    int32, per-batch start in flat layout
    B_Ready_Cache_Len,  # (batch,)    int32, per-batch prompt-cache length
    B_Q_Seq_Len,  # (batch,)    int32, per-batch new-token count
    B_Image_Token_End,  # (sum_q,)    int32, output scatter target
    BLOCK_SIZE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cache_len = tl.load(B_Ready_Cache_Len + cur_batch)
    q_seq_len = tl.load(B_Q_Seq_Len + cur_batch)
    image_num = tl.load(B_Image_Nums + cur_batch)
    image_start_num = tl.load(B_Image_Start_Num + cur_batch)
    flat_start = tl.load(B_Q_Start_Loc + cur_batch)

    for i in range(image_num):
        image_start_idx = tl.load(B_Image_Start_Idx + image_start_num + i)
        image_len = tl.load(B_Image_Len + image_start_num + i)
        image_end_idx = image_start_idx + image_len
        # Flat layout offset of the image's first token within this batch.
        flat_image_start = flat_start + image_start_idx - cache_len

        for j in range(0, image_len, BLOCK_SIZE):
            off = j + tl.arange(0, BLOCK_SIZE)
            in_image = off < image_len
            # Only fill positions that fall inside this batch's NEW-tokens range
            # (i.e., the part of the image that hasn't already been processed
            # in a previous chunked-prefill chunk and isn't past the chunk's end).
            in_new_tokens = (image_start_idx - cache_len + off >= 0) & (image_start_idx - cache_len + off < q_seq_len)
            tl.store(
                B_Image_Token_End + flat_image_start + off,
                image_end_idx,
                mask=in_image & in_new_tokens,
            )


def build_b_image_token_end(
    b_image_start_idx: torch.Tensor,
    b_image_len: torch.Tensor,
    b_image_nums: torch.Tensor,
    b_image_start_num: torch.Tensor,
    b_q_start_loc: torch.Tensor,
    b_ready_cache_len: torch.Tensor,
    b_q_seq_len: torch.Tensor,
    b_image_token_end: torch.Tensor,
):
    batch_size = b_q_start_loc.shape[0]
    assert b_image_nums.shape[0] == batch_size
    grid = (batch_size,)
    BLOCK_SIZE = 64
    _build_b_image_token_end_kernel[grid](
        b_image_start_idx,
        b_image_len,
        b_image_nums,
        b_image_start_num,
        b_q_start_loc,
        b_ready_cache_len,
        b_q_seq_len,
        b_image_token_end,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# ---------------------------------------------------------------------------
# Standalone correctness check
# ---------------------------------------------------------------------------


def _reference(
    multimodal_params,
    b_q_start_loc_cpu,
    b_ready_cache_len_cpu,
    b_q_seq_len_cpu,
    sum_q,
):
    out = torch.zeros((sum_q,), dtype=torch.int32)
    for batch_idx, params in enumerate(multimodal_params):
        cache_len = b_ready_cache_len_cpu[batch_idx]
        new_len = b_q_seq_len_cpu[batch_idx]
        flat_start = b_q_start_loc_cpu[batch_idx]
        for img in params.get("images", []):
            image_start_idx = img["start_idx"]
            image_end_idx = image_start_idx + img["token_num"]
            for j in range(img["token_num"]):
                req_off = image_start_idx - cache_len + j
                if req_off < 0 or req_off >= new_len:
                    continue
                out[flat_start + req_off] = image_end_idx
    return out


def _check():
    device = "cuda"
    # Two batches. b0 has 1 image overlapping new tokens; b1 has 2 images, one
    # fully cached and one in the new-token range.
    multimodal = [
        {"images": [{"start_idx": 5, "token_num": 4}]},  # b0: image at req[5..9)
        {
            "images": [
                {"start_idx": 0, "token_num": 3},  # fully cached
                {"start_idx": 8, "token_num": 5},  # in new tokens
            ]
        },
    ]
    b_q_start_loc = torch.tensor([0, 6], dtype=torch.int32)  # b0 new=6, b1 new=10
    b_ready_cache_len = torch.tensor([2, 5], dtype=torch.int32)
    b_q_seq_len = torch.tensor([6, 10], dtype=torch.int32)
    sum_q = int(b_q_seq_len.sum().item())

    ref = _reference(
        multimodal,
        b_q_start_loc.tolist(),
        b_ready_cache_len.tolist(),
        b_q_seq_len.tolist(),
        sum_q,
    )

    b_image_start_idx = []
    b_image_len = []
    b_image_nums = []
    b_image_start_num = []
    image_start_num = 0
    for params in multimodal:
        b_image_start_num.append(image_start_num)
        b_image_nums.append(len(params["images"]))
        for img in params["images"]:
            b_image_start_idx.append(img["start_idx"])
            b_image_len.append(img["token_num"])
            image_start_num += 1

    out_gpu = torch.zeros((sum_q,), dtype=torch.int32, device=device)
    build_b_image_token_end(
        b_image_start_idx=torch.tensor(b_image_start_idx, dtype=torch.int32, device=device),
        b_image_len=torch.tensor(b_image_len, dtype=torch.int32, device=device),
        b_image_nums=torch.tensor(b_image_nums, dtype=torch.int32, device=device),
        b_image_start_num=torch.tensor(b_image_start_num, dtype=torch.int32, device=device),
        b_q_start_loc=b_q_start_loc.to(device),
        b_ready_cache_len=b_ready_cache_len.to(device),
        b_q_seq_len=b_q_seq_len.to(device),
        b_image_token_end=out_gpu,
    )

    out_cpu = out_gpu.cpu()
    assert torch.equal(out_cpu, ref), f"\n got {out_cpu.tolist()}\n ref {ref.tolist()}"
    print("ok", out_cpu.tolist())


if __name__ == "__main__":
    if torch.cuda.is_available():
        _check()
    else:
        print("No CUDA, skip.")
