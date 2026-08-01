import torch
import itertools
import triton
import triton.language as tl
from typing import Optional
from lightllm.utils.log_utils import init_logger
from lightllm.common.triton_utils.autotuner import autotune

logger = init_logger(__name__)


def build_mrope_flat_idx(mrope_section, head_dim: int, device) -> torch.Tensor:
    if isinstance(mrope_section, torch.Tensor):
        section = [int(x) for x in mrope_section.tolist()]
    else:
        section = [int(x) for x in mrope_section]
    half = head_dim // 2
    idx = []
    start = 0
    for i, sf in enumerate([s * 2 for s in section]):
        plane = i % 3
        for d in range(start, start + sf):
            idx.append(plane * half + (d % half))
        start += sf
    return torch.tensor(idx, device=device, dtype=torch.int64)


@torch.no_grad()
def mrope_torch_ascend(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section,
    is_interleaved: bool = False,
    partial_rotary_factor: float = 1.0,
    flat_idx: Optional[torch.Tensor] = None,
) -> None:
    assert not is_interleaved, "interleaved mrope torch path is not implemented"
    assert partial_rotary_factor == 1.0, "partial_rotary_factor != 1.0 is not supported in mrope_torch"
    if isinstance(mrope_section, torch.Tensor):
        section = [int(x) for x in mrope_section.tolist()]
    else:
        section = [int(x) for x in mrope_section]
    head_dim = q.shape[-1]
    half = head_dim // 2
    assert cos.shape[-1] == half, f"expected cos last dim {half}, got {cos.shape[-1]}"

    if flat_idx is None:
        flat_idx = build_mrope_flat_idx(section, head_dim, q.device)
    tokens = cos.shape[1]
    stacked_c = cos.permute(1, 0, 2).reshape(tokens, 3 * half)
    stacked_s = sin.permute(1, 0, 2).reshape(tokens, 3 * half)
    cos_emb = stacked_c.index_select(1, flat_idx)
    sin_emb = stacked_s.index_select(1, flat_idx)
    import torch_npu

    q4 = q.unsqueeze(0).contiguous()
    k4 = k.unsqueeze(0).contiguous()
    c4 = cos_emb.unsqueeze(0).unsqueeze(2).contiguous()
    s4 = sin_emb.unsqueeze(0).unsqueeze(2).contiguous()
    q.copy_(torch_npu.npu_rotary_mul(q4, c4, s4, rotary_mode="half").squeeze(0))
    k.copy_(torch_npu.npu_rotary_mul(k4, c4, s4, rotary_mode="half").squeeze(0))


@triton.jit
def _mrope_triton_fused_kernel(
    q,
    k,
    Cos,
    Sin,
    mrope_section,
    stride_cosld,
    stride_cosd,
    stride_sinld,
    stride_sind,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    is_interleaved: tl.constexpr,
    HEAD_Q: tl.constexpr,
    HEAD_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    head_index = tl.program_id(0)
    seq_index = tl.program_id(1)

    dim_range0 = tl.arange(0, BLOCK_DMODEL // 2)
    dim_range1 = dim_range0 + BLOCK_DMODEL // 2

    t_cos = Cos + seq_index * stride_cosd
    h_cos = Cos + stride_cosld + seq_index * stride_cosd
    w_cos = Cos + 2 * stride_cosld + seq_index * stride_cosd
    t_sin = Sin + seq_index * stride_sind
    h_sin = Sin + stride_sinld + seq_index * stride_sind
    w_sin = Sin + 2 * stride_sinld + seq_index * stride_sind

    mrope_section_t = tl.load(mrope_section + 0)
    mrope_section_h = tl.load(mrope_section + 1)
    mrope_section_w = tl.load(mrope_section + 2)

    # Updated offsets for half head_dim
    offsets = tl.arange(0, BLOCK_DMODEL // 2)
    if is_interleaved:
        h_mask = ((offsets % 3) == 1) & (offsets <= 3 * mrope_section_h)
        w_mask = ((offsets % 3) == 2) & (offsets <= 3 * mrope_section_w)
        t_mask = ~(h_mask | w_mask)
    else:
        t_end = mrope_section_t
        h_end = t_end + mrope_section_h
        t_mask = offsets < mrope_section_t
        h_mask = (t_end <= offsets) & (offsets < h_end)
        w_mask = (h_end <= offsets) & (offsets < BLOCK_DMODEL // 2)

    t_cos = tl.load(t_cos + offsets, mask=t_mask, other=0)
    t_sin = tl.load(t_sin + offsets, mask=t_mask, other=0)
    h_cos = tl.load(h_cos + offsets, mask=h_mask, other=0)
    h_sin = tl.load(h_sin + offsets, mask=h_mask, other=0)
    w_cos = tl.load(w_cos + offsets, mask=w_mask, other=0)
    w_sin = tl.load(w_sin + offsets, mask=w_mask, other=0)

    cos = t_cos + h_cos + w_cos
    sin = t_sin + h_sin + w_sin

    if head_index < HEAD_Q:
        q_head_index = head_index
        off_q0 = seq_index * stride_qbs + q_head_index * stride_qh + dim_range0 * stride_qd
        off_q1 = seq_index * stride_qbs + q_head_index * stride_qh + dim_range1 * stride_qd
        q0 = tl.load(q + off_q0)
        q1 = tl.load(q + off_q1)
        out_q0 = q0 * cos - q1 * sin
        out_q1 = q0 * sin + q1 * cos
        tl.store(q + off_q0, out_q0)
        tl.store(q + off_q1, out_q1)
    else:
        k_head_index = head_index - HEAD_Q
        off_k0 = seq_index * stride_kbs + k_head_index * stride_kh + dim_range0 * stride_kd
        off_k1 = seq_index * stride_kbs + k_head_index * stride_kh + dim_range1 * stride_kd

        k0 = tl.load(k + off_k0)
        k1 = tl.load(k + off_k1)

        out_k0 = k0 * cos - k1 * sin
        out_k1 = k0 * sin + k1 * cos

        tl.store(k + off_k0, out_k0)
        tl.store(k + off_k1, out_k1)

    return


def get_test_configs():
    configs = []
    result = itertools.product([1, 2, 4, 8], [1, 2, 3, 4, 5])
    for num_warps, num_stages in result:
        t_config = {
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        configs.append(t_config)
    return configs


def get_static_key(q, k):
    head_num_q, head_num_k, head_dim = q.shape[1], k.shape[1], q.shape[2]
    return {
        "Q_HEAD_NUM": head_num_q,
        "K_HEAD_NUM": head_num_k,
        "HEAD_DIM": head_dim,
        "dtype": str(q.dtype),
    }


@autotune(
    kernel_name="mrope_triton_fused:v1",
    configs_gen_func=get_test_configs,
    static_key_func=get_static_key,
    run_key_func=lambda q: q.shape[0],
    mutates_args=["q", "k"],
)
@torch.no_grad()
def mrope_triton_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: torch.Tensor,
    is_interleaved: bool,
    partial_rotary_factor: float = 1.0,
    run_config: Optional[dict] = None,
):
    head_num_q, head_num_k = q.shape[1], k.shape[1]
    head_dim = int(q.shape[2] * partial_rotary_factor)
    num_tokens = q.shape[0]

    if not run_config:
        run_config = {"num_warps": 1, "num_stages": 1}

    num_stages = run_config["num_stages"]
    num_warps = run_config["num_warps"]

    grid = (head_num_q + head_num_k, num_tokens)
    _mrope_triton_fused_kernel[grid](
        q=q,
        k=k,
        Cos=cos,
        Sin=sin,
        mrope_section=mrope_section,
        stride_cosld=cos.stride(0),
        stride_cosd=cos.stride(1),
        stride_sinld=sin.stride(0),
        stride_sind=sin.stride(1),
        stride_qbs=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        stride_kbs=k.stride(0),
        stride_kh=k.stride(1),
        stride_kd=k.stride(2),
        is_interleaved=is_interleaved,
        HEAD_Q=head_num_q,
        HEAD_K=head_num_k,
        BLOCK_DMODEL=head_dim,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return
