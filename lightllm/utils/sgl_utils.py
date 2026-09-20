import torch
from typing import Optional, Tuple

from lightllm.common.triton_utils.autotuner import AutotuneKernelType, AutotuneLevel, Autotuner, autotune
from lightllm.utils.envs_utils import get_decode_attn_autotune_seq_len, get_triton_autotune_level
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


def _flash_attn_kvcache_run_key(page_table, max_seqlen_q, max_seqlen_k):
    batch_size = int(page_table.shape[0])
    max_q_len = int(max_seqlen_q)
    # 正常执行按调用方提供的真实 KV 长度查找配置，避免读取 CUDA 张量引入同步或使用 Graph 页表容量。
    max_kv_len = int(max_seqlen_k)
    # 只有开启 decode attention 调优时才使用重建输入的目标长度。
    if Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.DECODE_ATTENTION) and get_triton_autotune_level() in [
        AutotuneLevel.ADAPTIVE_AUTOTUNE,
        AutotuneLevel.FORCE_AUTOTUNE,
    ]:
        max_kv_len = get_decode_attn_autotune_seq_len()
    # run key 中的 KV 长度向上取整到 512 的整数倍，同一区间复用配置匹配结果，减少重复搜索。
    max_kv_len = (max_kv_len + 511) // 512 * 512
    return batch_size * 10_000_000_000_000 + max_q_len * 10_000_000 + max_kv_len


def _flash_attn_kvcache_rebuild_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    **kwargs,
):
    # 本回调在每次实际搜索配置前执行一次，输入构造开销不计入候选配置的 benchmark 耗时。
    # Graph 初始化时的真实请求通常很短，而页表宽度是 Graph 能容纳的最大 KV 长度，
    # 两者都不能直接代表期望调优的请求长度，因此需要单独构造用于计时的 KV 访问范围。
    # 返回的替换输入只用于本次调优；后续正常执行和 Graph 捕获仍使用调用方的原始输入。

    # 页表每行对应一个 attention 请求。MTP 下一个请求可能包含多个 query token，
    # 因此从页表取得请求数，不能直接把 Q 的 token 数当作 batch_size。
    batch_size = page_table.shape[0]

    # 用 LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN 指定代表性 KV 长度，单位是 token，默认 32768。
    # 调优阶段的 run_key 按同一个配置值分桶，保证配置对应实际 benchmark 的长度区间。
    kv_len = get_decode_attn_autotune_seq_len()

    # KV cache 布局为 [物理页数, 每页 token 数, KV head 数, head_dim]。
    # 将目标 token 数向上取整为所需页数，支持长度不整除 page_size，以及调优页表比原页表更宽或更窄。
    page_size = k_cache.shape[1]
    max_pages = (kv_len + page_size - 1) // page_size

    # K/V 共用同一张页表，物理页数必须一致；至少需要一页才能生成合法索引并进行后续取模。
    assert k_cache.shape[0] == v_cache.shape[0], "K/V caches must have the same number of pages"
    num_pages = k_cache.shape[0]
    if num_pages == 0:
        raise ValueError("FA3 autotuning requires a non-empty KV cache")

    # 复用模型已有的 K/V 存储，只分配较小的页表和长度张量，避免为每个模拟长请求另分配完整 KV。
    # 原页表可能只填充了占位请求所用的少量条目，不能仅增加 cache_seqlens 就直接读取后续条目。
    # 因此新建页表，并沿用原页表的 dtype/device；这里没有改写原页表或 K/V 数据。
    page_table = torch.arange(batch_size * max_pages, dtype=page_table.dtype, device=page_table.device)

    # 取模将页编号映射到 [0, num_pages)，再整理为每个请求一行、每行 max_pages 个页编号的页表。
    # 例如 batch_size=2、max_pages=4、num_pages=6，结果为 [[0, 1, 2, 3], [4, 5, 0, 1]]。
    # 物理页充足时，各请求使用不同页；不足时通过循环复用模拟多个长请求，避免越界。
    # 这种复用可能提高 GPU 缓存命中率，
    # 因此调优时的访存特征与各请求拥有独立 KV 的真实场景存在差异。
    page_table = page_table.remainder_(num_pages).view(batch_size, max_pages)

    # FA3 根据 cache_seqlens 决定每个请求实际读取多少个 KV token，因此所有模拟请求都设为目标长度。
    # 这里保留精确的 token 数，而不是取整后的页容量，最后一页的多余位置不会算入有效 KV 长度。
    # 新建 int32 长度张量，避免修改原始短请求的长度并影响后续 Graph 捕获。
    cache_seqlens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=k_cache.device)

    # Q、cu_seqlens_q 和 max_seqlen_q 保持原样，保留普通 decode、MTP 及不等长 query 分组的真实布局。
    # kwargs 原样透传 causal、window_size、softmax_scale 等设置，使计时采用与原调用一致的 attention 语义。
    # 返回参数顺序与 flash_attn_with_kvcache_autotune 的必填参数一致，仅替换 KV 页表和 KV 长度。
    return (q, k_cache, v_cache, cache_seqlens, page_table, cu_seqlens_q, max_seqlen_q, kv_len), kwargs


@autotune(
    kernel_name="sgl_fa3_kvcache_ns:v3",
    kernel_type=AutotuneKernelType.DECODE_ATTENTION,
    configs_gen_func=_flash_attn_kvcache_num_splits_configs,
    static_key_func=_flash_attn_kvcache_static_key,
    run_key_func=_flash_attn_kvcache_run_key,
    rebuild_input_func=_flash_attn_kvcache_rebuild_inputs,
)
@torch.no_grad()
def flash_attn_with_kvcache_autotune(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    num_splits: int = 0,
    sinks: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    return_softmax_lse: bool = False,
    run_config: Optional[dict] = None,
):
    # KV 长度、页表及 query 布局由调用方显式提供；四维 batched Q 的 cu_seqlens_q 可以显式传 None。
    # max_seqlen_k 是 CPU 上已知的真实最大 KV token 数，仅用于配置查找，不传给底层 FA3 算子。
    # 此场景的 KV 已提前写入缓存，FA3 只读取已有缓存，不追加新的 K/V。
    # cu_seqlens_k_new 仅描述新增 K/V 的分段，因此必须为 None；实际 KV 长度由 cache_seqlens 指定。
    assert cu_seqlens_k_new is None, "cu_seqlens_k_new must be None when autotuning attention over an existing KV cache"

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
        softmax_scale=softmax_scale,
        return_softmax_lse=return_softmax_lse,
    )
