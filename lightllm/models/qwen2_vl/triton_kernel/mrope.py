import time
import torch
import triton
import triton.language as tl


@triton.jit
def mrope_kernel(
    Q_ptr,
    K_ptr,
    COS_ptr,
    SIN_ptr,
    AXIS_ptr,
    QO_ptr,
    KO_ptr,
    B: tl.int32,
    H_q: tl.int32,
    H_k: tl.int32,
    L: tl.int32,
    D: tl.int32,
    HALF: tl.constexpr,
    s_tok: tl.int32,
    s_ax: tl.int32,
    s_d: tl.int32,
    q_sb: tl.int32,
    q_sh: tl.int32,
    q_sl: tl.int32,
    q_sd: tl.int32,
    k_sb: tl.int32,
    k_sh: tl.int32,
    k_sl: tl.int32,
    k_sd: tl.int32,
    qo_sb: tl.int32,
    qo_sh: tl.int32,
    qo_sl: tl.int32,
    qo_sd: tl.int32,
    ko_sb: tl.int32,
    ko_sh: tl.int32,
    ko_sl: tl.int32,
    ko_sd: tl.int32,
    BLOCK_D: tl.constexpr,
):

    total_h = H_q + H_k
    pid_bh = tl.program_id(0)
    pid_l = tl.program_id(1)

    b = pid_bh // total_h
    h_local = pid_bh - b * total_h

    is_q = h_local < H_q
    h_q = h_local
    h_k = h_local - H_q

    sb = tl.where(is_q, q_sb, k_sb)
    sh = tl.where(is_q, q_sh, k_sh)
    sl = tl.where(is_q, q_sl, k_sl)
    sd = tl.where(is_q, q_sd, k_sd)

    osb = tl.where(is_q, qo_sb, ko_sb)
    osh = tl.where(is_q, qo_sh, ko_sh)
    osl = tl.where(is_q, qo_sl, ko_sl)
    osd = tl.where(is_q, qo_sd, ko_sd)

    base_ptr = tl.where(is_q, Q_ptr, K_ptr)
    out_ptr = tl.where(is_q, QO_ptr, KO_ptr)
    h_index = tl.where(is_q, h_q, h_k)

    base = b * sb + h_index * sh + pid_l * sl
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    idx = base + offs * sd
    vals = tl.load(base_ptr + idx, mask=mask, other=0.0)

    rot_offs = tl.where(offs < HALF, (offs + HALF) * sd, (offs - HALF) * sd)
    rot_vals = tl.load(base_ptr + base + rot_offs, mask=mask, other=0.0)
    rot_vals = tl.where(offs < HALF, -rot_vals, rot_vals)

    axis_id = tl.load(AXIS_ptr + offs, mask=mask, other=0)  # 0,1,2
    idx_d = tl.where(offs < HALF, offs, offs - HALF)
    cos_idx = pid_l * s_tok + axis_id * s_ax + idx_d
    c = tl.load(COS_ptr + cos_idx, mask=idx_d < HALF, other=0.0)
    s = tl.load(SIN_ptr + cos_idx, mask=idx_d < HALF, other=0.0)

    out = vals * c + rot_vals * s

    out_idx = b * osb + h_index * osh + pid_l * osl + offs * osd
    tl.store(out_ptr + out_idx, out, mask=mask)


def mrope_triton(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, axis_map: torch.Tensor):

    B, H_q, L, D = q.shape
    H_k = k.shape[1]
    HALF = D // 2

    q_sb, q_sh, q_sl, q_sd = map(int, q.stride())
    k_sb, k_sh, k_sl, k_sd = map(int, k.stride())

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    qo_sb, qo_sh, qo_sl, qo_sd = map(int, q_out.stride())
    ko_sb, ko_sh, ko_sl, ko_sd = map(int, k_out.stride())

    assert len(cos.shape) == 3

    s_axis = int(cos.stride(0))
    s_token = int(cos.stride(1))
    s_d = int(cos.stride(2))

    grid = (B * (H_q + H_k), L)

    mrope_kernel[grid](
        q,
        k,
        cos,
        sin,
        axis_map,
        q_out,
        k_out,
        B,
        H_q,
        H_k,
        L,
        D,
        HALF,
        s_token,
        s_axis,
        s_d,
        q_sb,
        q_sh,
        q_sl,
        q_sd,
        k_sb,
        k_sh,
        k_sl,
        k_sd,
        qo_sb,
        qo_sh,
        qo_sl,
        qo_sd,
        ko_sb,
        ko_sh,
        ko_sl,
        ko_sd,
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=4,
        num_stages=3,
    )
    return q_out, k_out


# ----------------  test ---------------- #
def test():

    # torch实现
    def rotate_half(x: torch.Tensor):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
        chunks = mrope_section * 2
        cos_embed = torch.cat(
            [m[i % 3] for i, m in enumerate(cos.split(chunks, dim=-1))],
            dim=-1,
        ).unsqueeze(unsqueeze_dim)
        sin_embed = torch.cat(
            [m[i % 3] for i, m in enumerate(sin.split(chunks, dim=-1))],
            dim=-1,
        ).unsqueeze(unsqueeze_dim)

        q_out = q * cos_embed + rotate_half(q) * sin_embed
        k_out = k * cos_embed + rotate_half(k) * sin_embed
        return q_out, k_out

    B, H_q, H_k, L, D = 3, 28, 4, 16384, 128
    mrope_section = [16, 24, 24]
    torch.manual_seed(0)
    device = "cuda"

    q = torch.rand(B, H_q, L, D, dtype=torch.float32, device=device).transpose(1, 2).contiguous().transpose(1, 2)
    k = torch.rand(B, H_k, L, D, dtype=torch.float32, device=device).transpose(1, 2).contiguous().transpose(1, 2)
    cos = torch.rand(3, 1, L, D, dtype=torch.float32, device=device)
    sin = torch.rand(3, 1, L, D, dtype=torch.float32, device=device)

    # 精度对比
    axis_map = []
    for i, n in enumerate(mrope_section * 2):
        axis_map += [i % 3] * n
    axis_map = torch.tensor(axis_map, dtype=torch.int32, device="cuda")
    ref_q, ref_k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1)

    torch.cuda.synchronize()
    out_q, out_k = mrope_triton(q, k, cos, sin, axis_map)
    torch.cuda.synchronize()

    err_q = (out_q - ref_q).abs().max().item()
    err_k = (out_k - ref_k).abs().max().item()
    print(f"abs‑max error   q:{err_q:.6f}, k:{err_k:.6f}")

    assert err_q < 1e-2 and err_k < 1e-2

    # 速度对比
    n_iter = 100
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)

    e0.record()
    for _ in range(n_iter):
        _ = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1)
    e1.record()
    torch.cuda.synchronize()
    t_ref = e0.elapsed_time(e1) / n_iter

    e0.record()
    for _ in range(n_iter):
        _ = mrope_triton(q, k, cos, sin, axis_map)
    e1.record()
    torch.cuda.synchronize()
    t_tri = e0.elapsed_time(e1) / n_iter

    print(f"torch {t_ref:.2f} ms/iter")
    print(f"triton {t_tri:.2f} ms/iter")
