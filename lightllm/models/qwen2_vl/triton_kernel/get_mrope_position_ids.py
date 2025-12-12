import torch
import triton
import triton.language as tl


@triton.jit
def _mrope_pos_kernel(
    Input_ids,  # [L]
    Pos1d,  # [L]  (全局 1D 位置)
    Img_start_token_ids,  # [MAX_IMG]
    Img_token_lens,  # [MAX_IMG]
    Img_T,  # [MAX_IMG]
    Img_H,  # [MAX_IMG]
    Img_W,  # [MAX_IMG]
    Img_start_pos1d,  # [MAX_IMG]
    Img_span,  # [MAX_IMG]
    Img_base_scalar,  # [MAX_IMG]
    spatial_merge_size,  # int32 标量
    Out_pos,  # [L, 3]
    stride_out_s,
    stride_out_d,
    total_len,  # L
    img_count,  # 实际图片数 (<= MAX_IMG)
    MAX_IMG: tl.constexpr,  # 编译期常量
):

    token_idx = tl.program_id(0).to(tl.int64)
    if token_idx >= total_len:
        return

    token_id = tl.load(Input_ids + token_idx)
    pos1d = tl.load(Pos1d + token_idx)

    # 先按“文本逻辑 pos”来算第一维：减去前面 image token，加上 image span
    # 这里要对所有 img 做一次扫描
    # 注意：这个 pos_text 是“标量逻辑 pos”，后面如果是文本 token，就用它给三维
    pos_text = pos1d

    for img_idx in tl.static_range(0, MAX_IMG):
        valid = img_idx < img_count

        start_pos1d = tl.load(Img_start_pos1d + img_idx, mask=valid, other=0)
        token_len = tl.load(Img_token_lens + img_idx, mask=valid, other=0)
        span = tl.load(Img_span + img_idx, mask=valid, other=0)

        # 这张图的 token 区间是 [start_pos1d, start_pos1d + token_len)
        end_pos1d = start_pos1d + token_len

        finished = valid & (pos1d >= end_pos1d)

        # 如果当前 token 在这张图之后，它的文本逻辑 pos 要减 len_j，加 span_j
        delta = span - token_len
        pos_text = pos_text + tl.where(finished, delta, 0)

    # 默认先当文本 token：三维都是 pos_text
    pos_t = pos_text
    pos_h = pos_text
    pos_w = pos_text

    # 再看是不是图像 patch，如果是，用图片的 base + (t,h,w) 覆盖
    for img_idx in tl.static_range(0, MAX_IMG):
        valid = img_idx < img_count

        start_id = tl.load(Img_start_token_ids + img_idx, mask=valid, other=0)
        token_len = tl.load(Img_token_lens + img_idx, mask=valid, other=0)
        end_id = start_id + token_len

        in_img = valid & (token_id > start_id) & (token_id <= end_id)

        # 图内线性下标 k
        k = token_id - start_id - 1

        grid_H = tl.load(Img_H + img_idx, mask=valid, other=1)
        grid_W = tl.load(Img_W + img_idx, mask=valid, other=1)

        Hm = grid_H // spatial_merge_size
        Wm = grid_W // spatial_merge_size
        patch_per_frame = Hm * Wm

        t = k // patch_per_frame
        rest = k % patch_per_frame
        h = rest // Wm
        w = rest % Wm

        base_scalar = tl.load(Img_base_scalar + img_idx, mask=valid, other=0)

        new_pos_t = base_scalar + t
        new_pos_h = base_scalar + h
        new_pos_w = base_scalar + w

        pos_t = tl.where(in_img, new_pos_t, pos_t)
        pos_h = tl.where(in_img, new_pos_h, pos_h)
        pos_w = tl.where(in_img, new_pos_w, pos_w)

    # 写回 (L,3)
    tl.store(
        Out_pos + stride_out_s * token_idx + 0 * stride_out_d,
        pos_t,
    )
    tl.store(
        Out_pos + stride_out_s * token_idx + 1 * stride_out_d,
        pos_h,
    )
    tl.store(
        Out_pos + stride_out_s * token_idx + 2 * stride_out_d,
        pos_w,
    )
    return


@torch.no_grad()
def prepare_image_pos_meta_for_batch(
    seq_input_ids: torch.Tensor,
    seq_pos1d: torch.Tensor,
    images: list,
    spatial_merge_size: int = 2,
):
    device = seq_input_ids.device

    img_start_token_ids = []
    img_token_lens = []
    img_grid_thw = []
    img_start_pos1d = []
    img_span = []
    img_base_scalar = []

    if len(images) == 0:
        return (
            torch.empty(0, device=device, dtype=torch.long),  # start_token_ids
            torch.empty(0, device=device, dtype=torch.long),  # token_lens
            torch.empty(0, 3, device=device, dtype=torch.long),  # grid_thw
            torch.empty(0, device=device, dtype=torch.long),  # start_pos1d
            torch.empty(0, device=device, dtype=torch.long),  # span
            torch.empty(0, device=device, dtype=torch.long),  # base_scalar
        )

    # 先按 images 的顺序来，默认就是按序列出现顺序
    for img in images:
        start_id = int(img["token_id"])
        token_len = int(img["token_num"])
        grid_T, grid_H, grid_W = img["grid_thw"]
        grid_T = int(grid_T)
        grid_H = int(grid_H)
        grid_W = int(grid_W)

        Hm = grid_H // spatial_merge_size
        Wm = grid_W // spatial_merge_size
        num_patches = grid_T * Hm * Wm
        assert num_patches == token_len

        # 找这张图在当前 chunk 里第一次出现的 patch 位置
        # patch token id 区间是 [start_id, start_id + token_len)
        ids = seq_input_ids
        is_this_img = (ids >= start_id) & (ids < start_id + token_len)
        locs = torch.nonzero(is_this_img, as_tuple=False)
        if locs.numel() == 0:
            # 这张图的 patch 不在当前 chunk 里（可能在前一个或下一个 chunk），跳过
            # 注意：要保证跨 chunk 时在 infer_state 里缓存这些信息，这里简单起见先略过
            continue

        first_idx = int(locs[0].item())
        first_pos1d = int(seq_pos1d[first_idx].item())

        # 这张图在 Mrope 第一维上的占用长度：max(grid_T, Hm, Wm)
        span = max(grid_T, Hm, Wm)

        img_start_token_ids.append(start_id)
        img_token_lens.append(token_len)
        img_grid_thw.append([grid_T, grid_H, grid_W])
        img_start_pos1d.append(first_pos1d)
        img_span.append(span)

    if len(img_start_token_ids) == 0:
        return (
            torch.empty(0, device=device, dtype=torch.long),
            torch.empty(0, device=device, dtype=torch.long),
            torch.empty(0, 3, device=device, dtype=torch.long),
            torch.empty(0, device=device, dtype=torch.long),
            torch.empty(0, device=device, dtype=torch.long),
            torch.empty(0, device=device, dtype=torch.long),
        )

    img_start_token_ids = torch.tensor(img_start_token_ids, device=device, dtype=torch.long)
    img_token_lens = torch.tensor(img_token_lens, device=device, dtype=torch.long)
    img_grid_thw = torch.tensor(img_grid_thw, device=device, dtype=torch.long)
    img_start_pos1d = torch.tensor(img_start_pos1d, device=device, dtype=torch.long)
    img_span = torch.tensor(img_span, device=device, dtype=torch.long)

    # 对图片按 start_pos1d 排序，确保顺序和序列一致
    order = torch.argsort(img_start_pos1d)
    img_start_token_ids = img_start_token_ids[order]
    img_token_lens = img_token_lens[order]
    img_grid_thw = img_grid_thw[order]
    img_start_pos1d = img_start_pos1d[order]
    img_span = img_span[order]

    # 计算每张图的 base_scalar：
    # base_j = text_pos_after_前面所有文本和图片
    # 简单一点：用 “按 Mrope 压缩后的逻辑 pos” 来定义 base
    # 先算每张图前面一共出现了多少 image token 和多少 image span
    num_img = img_start_pos1d.shape[0]
    cum_token = torch.zeros(num_img, device=device, dtype=torch.long)
    cum_span = torch.zeros(num_img, device=device, dtype=torch.long)
    if num_img > 1:
        cum_token[1:] = img_token_lens.cumsum(0)[:-1]
        cum_span[1:] = img_span.cumsum(0)[:-1]

    # 对每张图，前面的文本 token 数约等于：img_start_pos1d - 总 image token 数（按 pos1d 是所有 token）
    # 所以它的逻辑起点可以近似：
    # base_j = (img_start_pos1d_j
    #           - cum_token_j      # 去掉前面图片多出来的 token
    #           + cum_span_j)      # 加回前面图片在 Mrope 里的 span
    img_base_scalar = img_start_pos1d - cum_token + cum_span

    return (
        img_start_token_ids,
        img_token_lens,
        img_grid_thw,
        img_start_pos1d,
        img_span,
        img_base_scalar,
    )


@torch.no_grad()
def gen_mrope_pos_triton(
    seq_input_ids: torch.Tensor,
    seq_pos1d: torch.Tensor,
    images: list,
    spatial_merge_size: int = 2,
    max_img_num: int = 128,
) -> torch.Tensor:
    device = seq_input_ids.device
    L = seq_input_ids.shape[0]

    (
        img_start_token_ids,
        img_token_lens,
        img_grid_thw,
        img_start_pos1d,
        img_span,
        img_base_scalar,
    ) = prepare_image_pos_meta_for_batch(seq_input_ids, seq_pos1d, images, spatial_merge_size=spatial_merge_size)

    n_img = img_start_token_ids.shape[0]
    assert n_img <= max_img_num

    img_start_token_ids = img_start_token_ids.to(device=device, dtype=torch.long).contiguous()
    img_token_lens = img_token_lens.to(device=device, dtype=torch.long).contiguous()
    img_grid_thw = img_grid_thw.to(device=device, dtype=torch.long).contiguous()
    img_start_pos1d = img_start_pos1d.to(device=device, dtype=torch.long).contiguous()
    img_span = img_span.to(device=device, dtype=torch.long).contiguous()
    img_base_scalar = img_base_scalar.to(device=device, dtype=torch.long).contiguous()

    Img_T = img_grid_thw[:, 0]
    Img_H = img_grid_thw[:, 1]
    Img_W = img_grid_thw[:, 2]

    # pad 到 max_img_num
    if n_img < max_img_num:
        pad = max_img_num - n_img
        zero = torch.zeros(pad, device=device, dtype=torch.long)
        one = torch.ones(pad, device=device, dtype=torch.long)

        img_start_token_ids = torch.cat([img_start_token_ids, zero], dim=0)
        img_token_lens = torch.cat([img_token_lens, zero], dim=0)
        Img_T = torch.cat([Img_T, one], dim=0)
        Img_H = torch.cat([Img_H, one], dim=0)
        Img_W = torch.cat([Img_W, one], dim=0)
        img_start_pos1d = torch.cat([img_start_pos1d, zero], dim=0)
        img_span = torch.cat([img_span, zero], dim=0)
        img_base_scalar = torch.cat([img_base_scalar, zero], dim=0)

    out = torch.empty((L, 3), device=device, dtype=torch.long)
    MAX_IMG = triton.next_power_of_2(n_img)
    grid = (L,)
    _mrope_pos_kernel[grid](
        seq_input_ids,
        seq_pos1d,
        img_start_token_ids,
        img_token_lens,
        Img_T,
        Img_H,
        Img_W,
        img_start_pos1d,
        img_span,
        img_base_scalar,
        spatial_merge_size,
        out,
        out.stride(0),
        out.stride(1),
        total_len=L,
        img_count=n_img,
        MAX_IMG=MAX_IMG,
        num_warps=1,
        num_stages=1,
    )

    return out.transpose(0, 1)  # (3, L)
