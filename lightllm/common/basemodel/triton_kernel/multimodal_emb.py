import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Prompt_ids, 
    Text_weight_embs,
    Img_embs,
    Out,
    Img_token_lens,
    Img_start_token_ids,
    Img_start_locs,
    stride_text_emb_s, stride_text_emb_d, # text_stride
    stride_img_emb_s, stride_img_emb_d, # img_stride
    stride_out_s, stride_out_d,
    tp_text_start_token_id,
    tp_text_end_token_id,
    hidden_size,
    BLOCK_HIDDEN_DIM: tl.constexpr
    ):

    seq_index = tl.program_id(0).to(tl.int64)
    img_handle_id = tl.program_id(1)

    token_id = tl.load(Prompt_ids + seq_index)
    off_d = tl.arange(0, BLOCK_HIDDEN_DIM)
    
    # load store text emb
    for _ in range(0, tl.where((img_handle_id == 0) & (token_id < tp_text_end_token_id) & (token_id >= tp_text_start_token_id), 1, 0), 1):
        load_emb = tl.load(Text_weight_embs + stride_text_emb_s * (token_id - tp_text_start_token_id) + off_d * stride_text_emb_d, mask=off_d < hidden_size, other=0)
        tl.store(Out + stride_out_s * seq_index + stride_out_d * off_d, load_emb, mask=off_d < hidden_size)
    
    img_start_token_id = tl.load(Img_start_token_ids + img_handle_id - 1, mask=img_handle_id >= 1, other=0)
    img_start_loc = tl.load(Img_start_locs + img_handle_id - 1, mask=img_handle_id >= 1, other=0)
    img_token_len = tl.load(Img_token_lens + img_handle_id - 1, mask=img_handle_id >= 1, other=0)
    # load store img emb
    for _ in range(0, tl.where((img_handle_id != 0) & (token_id >= img_start_token_id) & (token_id < img_start_token_id + img_token_len), 1, 0), 1):
        load_emb = tl.load(Img_embs + stride_img_emb_s * (img_start_loc + token_id - img_start_token_id) + off_d * stride_img_emb_d, mask=off_d < hidden_size, other=0)
        tl.store(Out + stride_out_s * seq_index + stride_out_d * off_d, load_emb, mask=off_d < hidden_size)
    return


@torch.no_grad()
def multimodal_emb(out: torch.Tensor, prompt_ids: torch.Tensor, text_weight_embs: torch.Tensor, img_embs: torch.Tensor, 
                         img_token_lens: torch.Tensor, img_start_token_ids: torch.Tensor, img_start_locs: torch.Tensor, 
                         tp_text_start_token_id,
                         tp_text_end_token_id):
    total_len = prompt_ids.shape[0]
    BLOCK = triton.next_power_of_2(out.shape[1])
    # print(len(img_token_lens))
    grid = (total_len, len(img_token_lens) + 1)
    num_warps = 1
    _fwd_kernel[grid](
        prompt_ids,
        text_weight_embs,
        img_embs,
        out,
        img_token_lens,
        img_start_token_ids,
        img_start_locs,
        text_weight_embs.stride(0), text_weight_embs.stride(1),
        img_embs.stride(0), img_embs.stride(1),
        out.stride(0), out.stride(1),
        tp_text_start_token_id,
        tp_text_end_token_id,
        hidden_size=out.shape[1],
        BLOCK_HIDDEN_DIM=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return



def test():
    S, D = 1024 * 1000, 128 * 64
    vob_size = 320000
    image_size = 10
    image_token_size = 512

    text_weight = torch.randn((vob_size, D), device='cuda', dtype=torch.float16)
    img_weight = torch.randn((image_size * image_token_size, D), device='cuda', dtype=torch.float16)
    img_token_lens = torch.full((image_size,), image_token_size, device='cuda', dtype=torch.long)
    img_start_token_ids = (torch.arange(0, image_size * image_token_size, image_token_size) + vob_size * 10).cuda().long()
    img_start_locs = torch.arange(0, image_size * image_token_size, image_token_size).cuda().long()

    prompt_ids = torch.arange(0, S, 1).cuda().long()
    prompt_ids[0: image_size * image_token_size] = (vob_size * 10 + torch.arange(0, image_size * image_token_size, 1)).cuda().long()

    out = torch.zeros((S, D), dtype=torch.float16, device="cuda")
    print(out.shape)

    import time
    
    triton_output = multimodal_emb(out, prompt_ids, text_weight, img_weight, img_token_lens, img_start_token_ids, img_start_locs, 0, vob_size)

    torch.cuda.synchronize()
    iters = 20
    t1 = time.time()
    for _ in range(iters):
        triton_output = multimodal_emb(out, prompt_ids, text_weight, img_weight, img_token_lens, img_start_token_ids, img_start_locs, 0, vob_size)
    torch.cuda.synchronize()
    t2 = time.time()
    print("Triton time cost", (t2 - t1) / iters)
    return


# if __name__ == "__main__":
#     test()

