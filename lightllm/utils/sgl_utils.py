import torch

from frozendict import frozendict
from lightllm.common.triton_utils.autotuner import AutotuneLevel, Autotuner
from lightllm.utils.envs_utils import get_triton_autotune_level
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

_DEFAULT_NUM_SPLITS = 0

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
    from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache as _flash_attn_with_kvcache

    flash_attn_varlen_func = flash_attn_varlen_func
    merge_state_v2 = sgl_ops.merge_state_v2
except:
    flash_attn_varlen_func = None
    _flash_attn_with_kvcache = None
    merge_state_v2 = None
    logger.warning(
        "sgl_kernel is not installed, or the installed version did not support fa3. \
        Try to upgrade it."
    )


def _flash_attn_kvcache_num_splits_configs():
    return [{"num_splits": num_splits} for num_splits in [0, 16, 32]]


def _flash_attn_kvcache_static_key(q, k_cache, v_cache, causal, window_size, softcap, sinks):
    return {
        "qd": str(q.dtype),
        "kd": str(k_cache.dtype),
        "vd": str(v_cache.dtype),
        "qh": int(q.shape[-2]),
        "kh": int(k_cache.shape[-2]),
        "hd": int(q.shape[-1]),
        "vh": int(v_cache.shape[-1]),
        "pb": int(k_cache.shape[-3]),
        "c": int(bool(causal)),
        "wl": int(window_size[0]),
        "wr": int(window_size[1]),
        "sc": int(softcap > 0.0),
        "sk": int(sinks is not None),
        "sgl": getattr(sgl_ops, "__version__", "unknown"),
    }


def _flash_attn_max_q_len(q, max_seqlen_q):
    if max_seqlen_q is not None:
        return int(max_seqlen_q)
    if q.dim() >= 4:
        return int(q.shape[1])
    return int(q.shape[0])


def _flash_attn_kvcache_run_key(q, page_table, max_seqlen_q):
    batch_size = int(page_table.shape[0])
    max_q_len = _flash_attn_max_q_len(q, max_seqlen_q)
    max_kv_len = int(page_table.shape[1])
    return batch_size * 1_000_000_000_000 + max_q_len * 1_000_000 + max_kv_len


def _flash_attn_is_decode_like(q, page_table, max_seqlen_q=None):
    if page_table is None or page_table.dim() < 2:
        return False

    max_q_len = _flash_attn_max_q_len(q, max_seqlen_q)
    if max_q_len <= 0 or int(page_table.shape[1]) <= max_q_len:
        return False

    q_token_num = int(q.shape[0]) * int(q.shape[1]) if q.dim() >= 4 else int(q.shape[0])
    return q_token_num == int(page_table.shape[0]) * max_q_len


def _flash_attn_should_autotune(q, kwargs):
    return (
        kwargs.get("num_splits", _DEFAULT_NUM_SPLITS) == _DEFAULT_NUM_SPLITS
        and kwargs.get("k") is None
        and kwargs.get("v") is None
        and kwargs.get("out") is None
        and kwargs.get("qv") is None
        and kwargs.get("q_descale") is None
        and kwargs.get("k_descale") is None
        and kwargs.get("v_descale") is None
        and _flash_attn_is_decode_like(q, kwargs.get("page_table"), kwargs.get("max_seqlen_q"))
    )


def _flash_attn_with_kvcache_autotune_call(call_kwargs):
    tuner = _flash_attn_with_kvcache_autotuned

    if get_triton_autotune_level() == AutotuneLevel.ADAPTIVE_AUTOTUNE:
        static_key = frozendict(tuner._static_key(**call_kwargs))
        run_key = str(tuner._run_key(**call_kwargs))
        tuner._try_load_cache(static_key)

        if run_key not in tuner.cached_configs.get(static_key, {}) and not Autotuner.is_autotune_warmup():
            Autotuner.start_autotune_warmup()
            try:
                return tuner(**call_kwargs)
            finally:
                Autotuner.end_autotune_warmup()

    return tuner(**call_kwargs)


def _flash_attn_decode_bench_kv_lens(page_table):
    if page_table is None or page_table.dim() < 2:
        return []

    max_kv_len = int(page_table.shape[1])
    if max_kv_len <= 0:
        return []

    return [10240, max_kv_len]


class _FlashAttnKvcacheAutotuner(Autotuner):
    def _bench(self, *args, n_repeat=3, n_retries=3, **kwargs):
        page_table = kwargs.get("page_table")
        cache_seqlens = kwargs.get("cache_seqlens")

        bench_times = []
        for bench_kv_len in _flash_attn_decode_bench_kv_lens(page_table):
            bench_kwargs = kwargs.copy()
            if isinstance(cache_seqlens, torch.Tensor):
                bench_cache_seqlens = cache_seqlens.clone()
                bench_cache_seqlens.fill_(bench_kv_len)
            else:
                bench_cache_seqlens = bench_kv_len
            bench_kwargs["cache_seqlens"] = bench_cache_seqlens

            cu_seqlens_k_new = bench_kwargs.get("cu_seqlens_k_new")
            if isinstance(cu_seqlens_k_new, torch.Tensor) and cu_seqlens_k_new.numel() != 0:
                bench_cu_seqlens_k_new = torch.arange(
                    cu_seqlens_k_new.numel(), device=cu_seqlens_k_new.device, dtype=cu_seqlens_k_new.dtype
                )
                bench_cu_seqlens_k_new *= bench_kv_len
                bench_kwargs["cu_seqlens_k_new"] = bench_cu_seqlens_k_new

            bench_times.append(super()._bench(*args, n_repeat=n_repeat, n_retries=n_retries, **bench_kwargs))

        if bench_times:
            return sum(bench_times) / len(bench_times)

        return super()._bench(*args, n_repeat=n_repeat, n_retries=n_retries, **kwargs)


if _flash_attn_with_kvcache is not None and torch.cuda.is_available():

    @torch.no_grad()
    def _flash_attn_with_kvcache_autotuned_impl(
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
        run_config=None,
        **kwargs,
    ):
        if run_config is not None:
            num_splits = run_config["num_splits"]
        return _flash_attn_with_kvcache(
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
            **kwargs,
        )

    _flash_attn_with_kvcache_autotuned = _FlashAttnKvcacheAutotuner(
        fn=_flash_attn_with_kvcache_autotuned_impl,
        kernel_name="sgl_fa3_kvcache_ns:v1",
        configs_gen_func=_flash_attn_kvcache_num_splits_configs,
        static_key_func=_flash_attn_kvcache_static_key,
        run_key_func=_flash_attn_kvcache_run_key,
    )

else:
    _flash_attn_with_kvcache_autotuned = None


def _flash_attn_with_kvcache_autotune_wrapper(q, k_cache, v_cache, **kwargs):
    if _flash_attn_with_kvcache_autotuned is None or not _flash_attn_should_autotune(q, kwargs):
        return _flash_attn_with_kvcache(q=q, k_cache=k_cache, v_cache=v_cache, **kwargs)

    call_kwargs = {"q": q, "k_cache": k_cache, "v_cache": v_cache, **kwargs}
    call_kwargs.setdefault("causal", False)
    call_kwargs.setdefault("window_size", (-1, -1))
    call_kwargs.setdefault("softcap", 0.0)
    call_kwargs.setdefault("sinks", None)
    call_kwargs.setdefault("num_splits", _DEFAULT_NUM_SPLITS)
    return _flash_attn_with_kvcache_autotune_call(call_kwargs)


flash_attn_with_kvcache = _flash_attn_with_kvcache
flash_attn_with_kvcache_autotune = (
    None if _flash_attn_with_kvcache is None else _flash_attn_with_kvcache_autotune_wrapper
)
