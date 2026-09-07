import torch

from lightllm.common.triton_utils.autotuner import AutotuneKernelType, autotune
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)
try:
    import sgl_kernel

    sgl_ops = sgl_kernel
    sgl_allreduce_ops = sgl_ops.allreduce
    HAS_SGL_KERNEL = True
except:
    sgl_ops = None
    sgl_allreduce_ops = None
    HAS_SGL_KERNEL = False
    logger.warning(
        "sgl_kernel is not installed, you can't use the api of it. \
                   You can solve it by running `pip install sgl_kernel`."
    )

try:
    from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    flash_attn_varlen_func = flash_attn_varlen_func
    flash_attn_with_kvcache = flash_attn_with_kvcache
    merge_state_v2 = sgl_ops.merge_state_v2
except:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
    merge_state_v2 = None
    logger.warning(
        "sgl_kernel is not installed, or the installed version did not support fa3. \
        Try to upgrade it."
    )


def _flash_attn_kvcache_num_splits_configs():
    return [{"num_splits": num_splits} for num_splits in [0, 16, 32]]


def _flash_attn_kvcache_static_key(q, k_cache, v_cache, causal, window_size, softcap, sinks, k_descale, v_descale):
    return {
        "qd": str(q.dtype),
        "kd": str(k_cache.dtype),
        "vd": str(v_cache.dtype),
        "qh": int(q.shape[-2]),
        "kh": int(k_cache.shape[-2]),
        "hd": int(q.shape[-1]),
        "vh": int(v_cache.shape[-1]),
        "pb": int(k_cache.shape[-3]),  # page size
        "c": int(bool(causal)),
        "wl": int(window_size[0]),
        "wr": int(window_size[1]),
        "sc": int(softcap > 0.0),
        "sk": int(sinks is not None),
        "has_k_descale": k_descale is not None,
        "has_v_descale": v_descale is not None,
        "sgl": getattr(sgl_ops, "__version__", "unknown"),
    }


def _flash_attn_max_q_len(q, max_seqlen_q):
    return int(max_seqlen_q if max_seqlen_q is not None else q.shape[1] if q.dim() >= 4 else q.shape[0])


def _flash_attn_kvcache_run_key(q, page_table, max_seqlen_q):
    batch_size = int(page_table.shape[0])
    max_q_len = _flash_attn_max_q_len(q, max_seqlen_q)
    max_kv_len = int(page_table.shape[1])
    return batch_size * 10_000_000_000_000 + max_q_len * 10_000_000 + max_kv_len


def _flash_attn_kvcache_rebuild_inputs(
    q,
    k_cache,
    v_cache,
    cache_seqlens=None,
    page_table=None,
    cu_seqlens_q=None,
    cu_seqlens_k_new=None,
    max_seqlen_q=None,
    *args,
    **kwargs,
):
    # Graph 初始化的占位 KV 长度通常只有 2，调优时按页表容量构造实际需要计算的长度。
    batch_size, max_pages = page_table.shape
    kv_len = max_pages * k_cache.shape[1]
    num_pages = min(k_cache.shape[0], v_cache.shape[0])
    if num_pages == 0:
        raise ValueError("FA3 autotuning requires a non-empty KV cache")

    # 复用已有 KV 存储，只重建页表；容量不足时循环使用合法物理页，不修改原始页表和 KV。
    page_table = torch.arange(batch_size * max_pages, dtype=page_table.dtype, device=page_table.device)
    page_table = page_table.remainder_(num_pages).view(batch_size, max_pages)
    cache_seqlens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=k_cache.device)
    if cu_seqlens_k_new is not None:
        cu_seqlens_k_new = torch.arange(batch_size + 1, dtype=torch.int32, device=k_cache.device) * kv_len

    # 保留原始 Q 和 query 分段，兼容普通 decode、MTP 及不同长度的 query 分组。
    return (q, k_cache, v_cache, cache_seqlens, page_table, cu_seqlens_q, cu_seqlens_k_new, max_seqlen_q, *args), kwargs


@autotune(
    kernel_name="sgl_fa3_kvcache_ns:v2",
    kernel_type=AutotuneKernelType.DECODE_ATTENTION,
    configs_gen_func=_flash_attn_kvcache_num_splits_configs,
    static_key_func=_flash_attn_kvcache_static_key,
    run_key_func=_flash_attn_kvcache_run_key,
    rebuild_input_func=_flash_attn_kvcache_rebuild_inputs,
)
@torch.no_grad()
def flash_attn_with_kvcache_autotune(
    q,
    k_cache,
    v_cache,
    cache_seqlens=None,
    page_table=None,
    cu_seqlens_q=None,
    cu_seqlens_k_new=None,
    max_seqlen_q=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    num_splits=0,
    sinks=None,
    k_descale=None,
    v_descale=None,
    run_config=None,
    **kwargs,
):
    if run_config is None:
        run_config = {"num_splits": 0}

    num_splits = run_config["num_splits"]
    return flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k_new,
        max_seqlen_q=max_seqlen_q,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        num_splits=num_splits,
        sinks=sinks,
        k_descale=k_descale,
        v_descale=v_descale,
        **kwargs,
    )
