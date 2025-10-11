import torch
import triton
import triton.language as tl
from lightllm.common.triton_utils.autotuner import autotune

# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
@triton.jit
def fused_gdn_gating_kernel(
    g,
    A_log,
    a,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)


def _get_fused_gdn_gating_configs():
    return [{"BLK_HEADS": bh, "num_warps": nw} for bh in [8, 16, 32, 64] for nw in [1, 2, 4]]


def _get_fused_gdn_gating_static_key(a: torch.Tensor):
    # group by head size and input dtype
    return {"NUM_HEADS": a.shape[1], "a_dtype": str(a.dtype)}


@autotune(
    kernel_name="fused_gdn_gating:v1",
    configs_gen_func=_get_fused_gdn_gating_configs,
    static_key_func=_get_fused_gdn_gating_static_key,
    run_key_func=lambda a: a.shape[0],
)
def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    run_config: dict | None = None,
) -> torch.Tensor:
    batch, num_heads = a.shape
    seq_len = 1

    # default heuristic when autotune is disabled
    if not run_config:
        # choose the largest block size that does not exceed num_heads
        candidate_blk = [8, 16, 32, 64]
        blk_heads = max([c for c in candidate_blk if c <= max(8, num_heads)] or [8])
        run_config = {"BLK_HEADS": blk_heads, "num_warps": 1}

    BLK_HEADS = run_config["BLK_HEADS"]
    num_warps = run_config.get("num_warps", 1)

    grid = (batch, seq_len, triton.cdiv(num_heads, BLK_HEADS))
    g = torch.empty_like(a, dtype=torch.float32)
    fused_gdn_gating_kernel[grid](
        g,
        A_log,
        a,
        dt_bias,
        seq_len,
        num_heads,
        beta,
        threshold,
        BLK_HEADS,
        num_warps=num_warps,
    )
    return g
