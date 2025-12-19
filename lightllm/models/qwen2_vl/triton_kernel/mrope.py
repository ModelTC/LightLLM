import time
import torch
import itertools
import triton
import triton.language as tl
from typing import Optional
from frozendict import frozendict
from functools import lru_cache
from lightllm.common.kernel_config import KernelConfigs
from lightllm.utils.log_utils import init_logger
from lightllm.common.triton_utils.autotuner import autotune

logger = init_logger(__name__)


class MropeTritonFusedKernelConfig(KernelConfigs):
    kernel_name: str = "mrope_triton_fused_kernel"

    @classmethod
    @lru_cache(maxsize=200)
    def try_to_get_best_config(
        cls,
        M: int,
        Q_HEAD_NUM: int,
        K_HEAD_NUM: int,
        HEAD_DIM: int,
        dtype: str,
    ) -> dict:
        key_params = {
            "Q_HEAD_NUM": Q_HEAD_NUM,
            "K_HEAD_NUM": K_HEAD_NUM,
            "HEAD_DIM": HEAD_DIM,
            "dtype": str(dtype),
        }
        key_params = frozendict(key_params)

        finded_config = cls.get_the_config(key_params)

        if finded_config:
            config = finded_config[min(finded_config.keys(), key=lambda x: abs(int(x) - M))]
            return config
        else:
            if M <= 256:
                config = {"num_warps": 1, "num_stages": 1}
            else:
                config = {"num_warps": 1, "num_stages": 1}

        return config

    @classmethod
    def save_config(
        cls,
        Q_HEAD_NUM: int,
        K_HEAD_NUM: int,
        HEAD_DIM: int,
        dtype: str,
        config_json: dict,
    ):
        key_params = {
            "Q_HEAD_NUM": Q_HEAD_NUM,
            "K_HEAD_NUM": K_HEAD_NUM,
            "HEAD_DIM": HEAD_DIM,
            "dtype": str(dtype),
        }
        key_params = frozendict(key_params)

        return cls.store_config(key_params, config_json)


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
    run_config: Optional[dict] = None,
):
    head_num_q, head_num_k = q.shape[1], k.shape[1]
    head_dim = int(q.shape[2])
    num_tokens = q.shape[0]

    if not run_config:
        run_config = MropeTritonFusedKernelConfig.try_to_get_best_config(
            M=num_tokens,
            Q_HEAD_NUM=head_num_q,
            K_HEAD_NUM=head_num_k,
            HEAD_DIM=head_dim,
            dtype=str(q.dtype),
        )
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
