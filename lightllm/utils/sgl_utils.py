import torch

from frozendict import frozendict
from lightllm.common.triton_utils.autotuner import AutotuneLevel, Autotuner
from lightllm.utils.envs_utils import get_triton_autotune_level
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

    flash_attn_with_kvcache_autotune = flash_attn_with_kvcache
    merge_state_v2 = sgl_ops.merge_state_v2
except:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
    flash_attn_with_kvcache_autotune = None
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
    return int(max_seqlen_q if max_seqlen_q is not None else q.shape[1] if q.dim() >= 4 else q.shape[0])


def _flash_attn_kvcache_run_key(q, page_table, max_seqlen_q):
    batch_size = int(page_table.shape[0])
    max_q_len = _flash_attn_max_q_len(q, max_seqlen_q)
    max_kv_len = int(page_table.shape[1])
    return batch_size * 1_000_000_000_000 + max_q_len * 1_000_000 + max_kv_len


class _FlashAttnKvcacheAutotuner(Autotuner):
    def _bench(self, *args, n_repeat=3, n_retries=3, **kwargs):
        page_table = kwargs.get("page_table")
        cache_seqlens = kwargs.get("cache_seqlens")
        max_kv_len = int(page_table.shape[1])

        bench_times = []
        for bench_kv_len in sorted({kv_len for kv_len in [10240, max_kv_len] if 0 < kv_len <= max_kv_len}):
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
                    cu_seqlens_k_new.numel(),
                    device=cu_seqlens_k_new.device,
                    dtype=cu_seqlens_k_new.dtype,
                )
                bench_cu_seqlens_k_new *= bench_kv_len
                bench_kwargs["cu_seqlens_k_new"] = bench_cu_seqlens_k_new

            bench_times.append(super()._bench(*args, n_repeat=n_repeat, n_retries=n_retries, **bench_kwargs))

        return sum(bench_times) / len(bench_times)


_flash_attn_with_kvcache_autotuned = None

if flash_attn_with_kvcache is not None and torch.cuda.is_available():

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
            **kwargs,
        )

    _flash_attn_with_kvcache_autotuned = _FlashAttnKvcacheAutotuner(
        fn=_flash_attn_with_kvcache_autotuned_impl,
        kernel_name="sgl_fa3_kvcache_ns:v1",
        configs_gen_func=_flash_attn_kvcache_num_splits_configs,
        static_key_func=_flash_attn_kvcache_static_key,
        run_key_func=_flash_attn_kvcache_run_key,
    )

    def _flash_attn_with_kvcache_autotune(q, k_cache, v_cache, **kwargs):
        tuner = _flash_attn_with_kvcache_autotuned
        call_kwargs = {"q": q, "k_cache": k_cache, "v_cache": v_cache, **kwargs}
        call_kwargs.setdefault("causal", False)
        call_kwargs.setdefault("window_size", (-1, -1))
        call_kwargs.setdefault("softcap", 0.0)
        call_kwargs.setdefault("sinks", None)

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

    flash_attn_with_kvcache_autotune = _flash_attn_with_kvcache_autotune


def fa3_decode_autotune(model, cuda_graph_batch_sizes):
    if _flash_attn_with_kvcache_autotuned is None or get_triton_autotune_level() not in [
        AutotuneLevel.ADAPTIVE_AUTOTUNE,
        AutotuneLevel.FORCE_AUTOTUNE,
    ]:
        return

    decode_backends = [
        model.decode_att_backend,
        getattr(model, "decode_att_backend1", None),
    ]
    if not any(backend is not None and backend.__class__.__name__ == "Fa3AttBackend" for backend in decode_backends):
        return

    need_end_warmup = not Autotuner.is_autotune_warmup()
    if need_end_warmup:
        Autotuner.start_autotune_warmup()
    try:
        max_kv_len = int(model.graph_max_len_in_batch)
        if max_kv_len <= 0:
            return

        k, v = model.mem_manager.get_att_input_params(layer_index=0)
        k_cache = k.view(k.shape[0], 1, k.shape[1], k.shape[2])
        v_cache = v.view(v.shape[0], 1, v.shape[1], v.shape[2])
        q_head_num = int(model.config["num_attention_heads"]) // model.tp_world_size_
        head_dim = int(k.shape[-1])
        mtp_size = model.args.mtp_step + 1
        hold_token_memindex = model.mem_manager.HOLD_TOKEN_MEMINDEX
        k[hold_token_memindex].zero_()
        v[hold_token_memindex].zero_()

        for batch_size in cuda_graph_batch_sizes[::-1]:
            att_batch_size = batch_size // mtp_size
            if att_batch_size <= 0:
                continue

            q = torch.zeros(
                (att_batch_size * mtp_size, q_head_num, head_dim),
                dtype=model.data_type,
                device=k.device,
            )
            page_table = torch.full(
                (att_batch_size, max_kv_len),
                hold_token_memindex,
                dtype=torch.int32,
                device=k.device,
            )
            cache_seqlens = torch.full((att_batch_size,), max_kv_len, dtype=torch.int32, device=k.device)
            cu_seqlens_q = torch.arange(att_batch_size + 1, dtype=torch.int32, device=k.device) * mtp_size
            cu_seqlens_k = torch.arange(att_batch_size + 1, dtype=torch.int32, device=k.device) * max_kv_len
            softmax_scale = 1.0 / (head_dim ** 0.5)

            flash_attn_with_kvcache_autotune(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_new=cu_seqlens_k,
                max_seqlen_q=mtp_size,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=(-1, -1),
                softcap=0.0,
                k_descale=None,
                v_descale=None,
                return_softmax_lse=False,
                sinks=None,
            )
    finally:
        if need_end_warmup:
            Autotuner.end_autotune_warmup()
    return
