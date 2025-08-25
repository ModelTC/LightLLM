import triton
import orjson
import os
from pathlib import Path
from tqdm import tqdm
import inspect
import torch
import torch.distributed as dist
from functools import lru_cache
from lightllm.utils.device_utils import get_current_device_name
import math
import fcntl
from lightllm.utils.log_utils import init_logger
import traceback

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def get_triton_version():
    return f"triton_{triton.__version__}"


def split_configs(configs):
    from lightllm.utils.dist_utils import get_current_rank_in_node, get_node_world_size
    import random

    random.shuffle(configs)
    rank_in_node = get_current_rank_in_node()
    node_world_size = get_node_world_size()
    return configs[rank_in_node::node_world_size]


def dict_to_filename(data):
    parts = []
    for k, v in sorted(data.items(), key=lambda x: str(x[0])):
        safe_k = str(k).replace(" ", "_").replace(":", "_")
        safe_v = str(v).replace(" ", "_").replace(":", "_")
        parts.append(f"{safe_k}={safe_v}")
    return ",".join(parts)


def nearest_power_of_2(x):
    # 返回最接近 x 的 2 的幂次方
    if x <= 1:
        return 1
    return triton.next_power_of_2(x - triton.next_power_of_2(x) // 4)


class BenchmarkState:
    def __init__(self):
        self.sum = 0
        self.min = float("inf")
        self.avg = 0
        self.count = 0

    def update(self, measurement):
        self.sum += measurement
        self.min = min(self.min, measurement)
        self.count += 1
        self.avg = self.sum / self.count


class Autotuner:
    @staticmethod
    def _get_param_names(func):
        if not callable(func):
            return None
        try:
            sig = inspect.signature(func)
            return [
                name
                for name, p in sig.parameters.items()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            ]
        except (ValueError, TypeError):
            return None

    def __init__(
        self,
        fn,
        arg_names,
        name,
        configs,
        default_config,
        static_key_func,
        run_key_func,
        reset_to_zero,
        restore_value,
        pre_hook=None,
        post_hook=None,
        warmup=None,
        rep=None,
    ):

        self.print_autotune = os.environ.get("LIGHTLLM_TRITON_PRINT_AUTOTUNE", "0") == "1"
        self.configs = configs
        self.default_config = default_config
        self.name = name
        self.cache_dir = os.path.join(
            Path(__file__).parent, "all_kernel_configs", get_triton_version(), get_current_device_name(), self.name
        )
        self.fn = fn
        self.static_key_func = static_key_func
        self.run_key_func = run_key_func

        self.cached_configs = {}
        self.arg_names = arg_names
        self._argname_to_pos = {name: idx for idx, name in enumerate(self.arg_names)}

        self._static_param_names = self._get_param_names(self.static_key_func)
        self._run_param_names = self._get_param_names(self.run_key_func)

        self.reset_to_zero = []
        if reset_to_zero is not None:
            self.reset_to_zero = list(reset_to_zero)
        self.restore_value = []
        if restore_value is not None:
            self.restore_value = list(restore_value)

        self.pre_hook = lambda kwargs, reset_only=False: 0
        self.post_hook = lambda kwargs, exception: 0
        self.user_defined_pre_hook = False
        self.user_defined_post_hook = False
        if pre_hook:
            self.pre_hook = pre_hook
            self.user_defined_pre_hook = True
        elif len(self.reset_to_zero) > 0 or len(self.restore_value) > 0:

            def _pre_hook(kwargs, reset_only=False):
                for name in self.reset_to_zero:
                    kwargs[name].zero_()
                if not reset_only:
                    self.restore_copies = {name: kwargs[name].clone() for name in self.restore_value}

            self.pre_hook = _pre_hook

        if post_hook:
            self.post_hook = post_hook
            self.user_defined_post_hook = True
        elif len(self.restore_value) > 0:

            def _post_hook(kwargs, exception):
                for name in self.restore_value:
                    kwargs[name].copy_(self.restore_copies[name])
                self.restore_copies = {}

            self.post_hook = _post_hook

        if not os.path.exists(self.cache_dir):
            if os.environ.get("LIGHTLLM_TRITON_AUTOTUNE", "0") == "1":
                os.makedirs(self.cache_dir, exist_ok=True)

        self._loaded_static_keys = set()
        self.sorted_cached_configs = {}
        self.early_stop_cnt = 0

    @lru_cache(maxsize=None)
    def _ensure_cache_loaded(self, static_key: str):
        if static_key in self._loaded_static_keys:
            return
        cache_file = os.path.join(self.cache_dir, f"{static_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    self.cached_configs[static_key] = orjson.loads(f.read())
                    self.sorted_cached_configs[static_key] = [
                        (int(k), v) for k, v in self.cached_configs[static_key].items()
                    ]
                    self.sorted_cached_configs[static_key].sort(key=lambda x: x[0])
            except Exception:
                # 若缓存损坏，忽略并在之后覆盖
                self.cached_configs[static_key] = {}
        self._loaded_static_keys.add(static_key)

    def _bench(self, *args, n_repeat=5, **kwargs):
        from triton.compiler.errors import CompileTimeAssertionFailure
        from triton.runtime.errors import OutOfResources, PTXASError

        full_nargs = {**self.nargs, **kwargs}

        def kernel_call():
            self.pre_hook(full_nargs)
            try:
                self.fn(*args, **kwargs)
            except Exception as e:
                try:
                    self.post_hook(full_nargs, exception=e)
                finally:
                    raise

            self.post_hook(full_nargs, exception=None)

        try:
            # warmup
            kernel_call()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(n_repeat):
                    kernel_call()
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            g.replay()
            end_event.record()
            torch.cuda.synchronize()
            return start_event.elapsed_time(end_event) / n_repeat

        except (OutOfResources, PTXASError, CompileTimeAssertionFailure, RuntimeError, Exception):
            return float("inf")

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        static_key = self._static_key(*args, **kwargs)
        run_key = self._run_key(*args, **kwargs)

        # 懒加载
        self._ensure_cache_loaded(static_key)
        best_config = None
        self.nargs = dict(zip(self.arg_names, args))

        def _benchmark(_run_key):
            from lightllm.utils.dist_utils import get_global_rank

            rank_id = get_global_rank()
            bar = tqdm(
                self.configs,
                desc=f"Autotuning {self.name} [rank:{rank_id}] for {_run_key}",
                position=get_global_rank(),
                dynamic_ncols=True,
            )
            _best_time = float("inf")
            run_time_list = []
            for i, config in enumerate(bar):
                kwargs_with_config = kwargs.copy()
                kwargs_with_config["run_config"] = config
                run_time = self._bench(*args, **kwargs_with_config)
                if run_time < _best_time:
                    _best_time = run_time
                run_time_list.append(run_time)
                bar.set_description(
                    f"Autotuning {self.name} [rank:{rank_id}] \
                        for {_run_key}, best_time: {_best_time:.5f}"
                )
            # 在所有设备上聚合每个配置的运行时间之和，并基于该总和选择最佳配置
            aggregated_time_list = run_time_list

            if dist.is_initialized():
                time_tensor = torch.tensor(
                    run_time_list,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                )
                dist.all_reduce(time_tensor, op=dist.ReduceOp.SUM)
                aggregated_time_list = time_tensor.detach().cpu().tolist()

            if len(aggregated_time_list) > 0:
                best_idx = min(range(len(aggregated_time_list)), key=lambda i: aggregated_time_list[i])
                _best_config = self.configs[best_idx]
            else:
                _best_config = self.default_config

            if static_key not in self.cached_configs:
                self.cached_configs[static_key] = {}
            self.cached_configs[static_key][run_key] = _best_config
            self.sorted_cached_configs[static_key] = [(int(k), v) for k, v in self.cached_configs[static_key].items()]
            self.sorted_cached_configs[static_key].sort(key=lambda x: x[0])

            if not dist.is_initialized() or get_global_rank() == 0:
                if os.environ.get("LIGHTLLM_TRITON_AUTOTUNE", "0") == "1":
                    cache_file = os.path.join(self.cache_dir, f"{static_key}.json")
                    with open(cache_file, "wb") as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        try:
                            f.write(
                                orjson.dumps(
                                    self.cached_configs[static_key], option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
                                )
                            )
                        finally:
                            fcntl.flock(f, fcntl.LOCK_UN)
                    if self.print_autotune:
                        logger.info(f"Saved configs for {self.name} - {static_key} - {run_key}")

            kwargs["run_config"] = self.cached_configs[static_key][run_key]
            full_nargs = {**self.nargs, **kwargs}
            self.pre_hook(full_nargs, reset_only=True)

        if static_key in self.cached_configs:
            if run_key in self.cached_configs[static_key]:
                best_config = self.cached_configs[static_key][run_key]

        if best_config is None:
            if os.environ.get("LIGHTLLM_TRITON_AUTOTUNE", "0") == "1":
                _benchmark(run_key)
            else:
                if static_key in self.sorted_cached_configs:
                    sorted_configs = self.sorted_cached_configs[static_key]
                    self.cached_configs[static_key][run_key] = min(
                        sorted_configs, key=lambda x: abs(x[0] - int(run_key))
                    )[1]
                else:
                    if static_key not in self.cached_configs:
                        logger.warning(
                            f"No kernel config for {self.name} in {self.cache_dir}/{static_key}, using default config"
                        )
                        self.cached_configs[static_key] = {}
                    self.cached_configs[static_key][run_key] = self.default_config

            best_config = self.cached_configs[static_key][run_key]

        kwargs["run_config"] = best_config
        return self.fn(*args, **kwargs)

    def _select_args(self, param_names, args, kwargs):
        if not param_names:
            return ()
        values = []
        for name in param_names:
            if name in kwargs:
                values.append(kwargs[name])
                continue
            pos = self._argname_to_pos.get(name, None)
            if pos is not None and pos < len(args):
                values.append(args[pos])
            else:
                raise KeyError(f"Missing argument '{name}' required by key function")
        return tuple(values)

    def _static_key(self, *args, **kwargs):
        if callable(self.static_key_func):
            try:
                params = self._select_args(self._static_param_names, args, kwargs)
                key = (
                    self.static_key_func(*params)
                    if self._static_param_names is not None
                    else self.static_key_func(*args, **kwargs)
                )
            except Exception:
                key = self.static_key_func(*args, **kwargs)
            if isinstance(key, dict):
                return dict_to_filename(key)
            return str(key)
        return "default"

    def _run_key(self, *args, **kwargs):
        if callable(self.run_key_func):
            try:
                params = self._select_args(self._run_param_names, args, kwargs)
                key = (
                    self.run_key_func(*params)
                    if self._run_param_names is not None
                    else self.run_key_func(*args, **kwargs)
                )
            except Exception:
                key = self.run_key_func(*args, **kwargs)
            if isinstance(key, dict):
                return dict_to_filename(key)
            return str(key)
        return "default"


def autotune(
    name,
    configs,
    default_config,
    static_key_func=None,
    run_key_func=None,
    reset_to_zero=None,
    restore_value=None,
    pre_hook=None,
    post_hook=None,
):
    def decorator(fn):
        arg_names = [param.name for param in inspect.signature(fn).parameters.values()]
        return Autotuner(
            fn,
            arg_names,
            name,
            configs,
            default_config,
            static_key_func,
            run_key_func,
            reset_to_zero,
            restore_value,
            pre_hook=pre_hook,
            post_hook=post_hook,
        )

    return decorator
