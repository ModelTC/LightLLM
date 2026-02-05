import torch

import triton
import triton.language as tl
import os


@triton.jit
def _rms_norm_fwd_fused(
    X,  # pointer to the input
    W,  # pointer to the weights
    x_stride0,  # how much to increase the pointer when moving by 1 row
    x_stride1,
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    head_idx = tl.program_id(1)

    X += row * x_stride0
    # Compute variance
    cols = (head_idx * head_dim + tl.arange(0, BLOCK_SIZE)) * x_stride1
    x = tl.load(X + cols).to(tl.float32)
    var = tl.sum(x * x, axis=0) / head_dim
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    w = tl.load(W + tl.arange(0, BLOCK_SIZE)).to(tl.float32)
    x_hat = x * rstd
    y = x_hat * w
    # Write output
    tl.store(X + cols, y.to(X.dtype.element_ty))


def qk_rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps):
    """
    This function is used to perform in-place RMSNorm on the input tensor,
    and to adapt the head_dim norm for Qwen3 MoE and the splited qk tensor layout.
    x: (M, N)
    weight: (head_dim,)
    eps: float
    return: x
    """
    assert weight.is_contiguous()
    # reshape input data into 2D tensor
    x_arg = x.view(-1, x.shape[-1])
    M, N = x_arg.shape
    head_dim = weight.shape[0]
    assert x.shape[-1] % head_dim == 0
    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    assert BLOCK_SIZE <= head_dim, "head_dim must be the power of 2"
    # enqueue kernel
    _rms_norm_fwd_fused[(M, N // head_dim)](
        x_arg,
        weight,
        x_arg.stride(0),
        x_arg.stride(1),
        N,
        eps,
        head_dim=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return x


@triton.jit
def _qk_rms_norm_fused_kernel(
    # Q Pointers & Strides
    Q_ptr,
    WQ_ptr,
    stride_q_row,
    stride_q_col,
    # K Pointers & Strides
    K_ptr,
    WK_ptr,
    stride_k_row,
    stride_k_col,
    # Dimensions
    num_heads_q: tl.constexpr,  # Q 的头数 (用于判断边界)
    head_dim: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # PID 0: 处理第几个 Token (Row)
    row_idx = tl.program_id(0)
    # PID 1: 处理第几个 Head (Combo Index)
    # 范围是 [0, num_heads_q + num_heads_k)
    combo_head_idx = tl.program_id(1)

    # 公共的 offset (0 ~ head_dim)
    offs = tl.arange(0, BLOCK_SIZE)

    # === 分支逻辑：判断是处理 Q 还是 K ===
    if combo_head_idx < num_heads_q:
        # ------------------ 处理 Q ------------------
        # 指针计算
        # Q 的实际 head index 就是 combo_head_idx
        Q_ptr += row_idx * stride_q_row

        # 定位 Q 数据: Base + Row偏移 + Head偏移 + 列偏移
        q_ptr_offset = (combo_head_idx * head_dim + offs) * stride_q_col

        # 加载 Q 数据
        x = tl.load(Q_ptr + q_ptr_offset).to(tl.float32)
        # RMSNorm 计算
        var = tl.sum(x * x, axis=0) / head_dim
        rstd = 1 / tl.sqrt(var + eps)

        # 加载 Q 的权重 (假设所有 Head 共享同一组 dim=head_dim 的权重)
        w = tl.load(WQ_ptr + offs).to(tl.float32)

        y = x * rstd * w

        # 写回 Q
        tl.store(Q_ptr + q_ptr_offset, y.to(Q_ptr.dtype.element_ty))

    else:
        # ------------------ 处理 K ------------------
        # 重新映射 K 的 head index (从 0 开始)
        k_head_idx = combo_head_idx - num_heads_q

        # 指针计算
        K_ptr += row_idx * stride_k_row
        k_ptr_offset = (k_head_idx * head_dim + offs) * stride_k_col

        # 加载 K 数据
        x = tl.load(K_ptr + k_ptr_offset).to(tl.float32)
        # RMSNorm 计算
        var = tl.sum(x * x, axis=0) / head_dim
        rstd = 1 / tl.sqrt(var + eps)

        # 加载 K 的权重
        w = tl.load(WK_ptr + offs).to(tl.float32)

        y = x * rstd * w

        # 写回 K
        tl.store(K_ptr + k_ptr_offset, y.to(K_ptr.dtype.element_ty))


def qk_rmsnorm_fused_forward(q: torch.Tensor, k: torch.Tensor, w_q: torch.Tensor, w_k: torch.Tensor, eps: float = 1e-6):
    """
    In-place RMSNorm for both Q and K in a single kernel launch.
    Supports GQA (different number of heads for Q and K).

    Args:
        q: (Total_Tokens, Hidden_Q) or (B, S, H_q, D) -> flattend to 2D inside
        k: (Total_Tokens, Hidden_K)
        w_q: (head_dim,) Scale parameter for Q
        w_k: (head_dim,) Scale parameter for K
    """
    # 1. 维度与连续性检查
    # 将输入统一视为 (Total_Tokens, Hidden_Size) 的 2D 视图
    q_view = q.view(-1, q.shape[-1])
    k_view = k.view(-1, k.shape[-1])

    assert w_q.is_contiguous() and w_k.is_contiguous()

    M = q_view.shape[0]  # Total Tokens
    assert k_view.shape[0] == M, "Q and K must have the same number of tokens"

    head_dim = w_q.shape[0]
    assert w_k.shape[0] == head_dim, "Head dim of Q and K must match"

    # 计算 Head 数量
    N_q = q_view.shape[1]
    N_k = k_view.shape[1]

    assert N_q % head_dim == 0
    assert N_k % head_dim == 0

    num_heads_q = N_q // head_dim
    num_heads_k = N_k // head_dim

    # 2. Block Size 设置
    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    assert BLOCK_SIZE == head_dim, "Currently only supports head_dim power of 2 (e.g., 64, 128)"

    # 3. 启动 Kernel
    # Grid: (Token数量, Q头数 + K头数)
    grid = (M, num_heads_q + num_heads_k)

    _qk_rms_norm_fused_kernel[grid](
        q_view,
        w_q,
        q_view.stride(0),
        q_view.stride(1),
        k_view,
        w_k,
        k_view.stride(0),
        k_view.stride(1),
        num_heads_q=num_heads_q,
        head_dim=head_dim,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    return q, k


# ================= 测试代码 =================
if __name__ == "__main__":
    torch.manual_seed(0)

    # 模拟配置: Batch=2, Seq=128, Head_Dim=128
    # Q: 16 Heads, K: 4 Heads (GQA 场景)
    B, S, D = 2, 128, 128
    H_Q = 16
    H_K = 4

    q = torch.randn((B * S, H_Q * D), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((B * S, H_K * D), device="cuda", dtype=torch.bfloat16)

    w_q = torch.ones((D,), device="cuda", dtype=torch.bfloat16)
    w_k = torch.ones((D,), device="cuda", dtype=torch.bfloat16)

    # 复制一份做对比
    q_ref = q.clone()
    k_ref = k.clone()

    # 执行 Fused Kernel
    qk_rmsnorm_fused_forward(q, k, w_q, w_k)

    q_ref_out = qk_rmsnorm_forward(q_ref, w_q, eps=1e-6)
    k_ref_out = qk_rmsnorm_forward(k_ref, w_k, eps=1e-6)

    print("Q Diff:", (q - q_ref_out).abs().max().item())
    print("K Diff:", (k - k_ref_out).abs().max().item())
