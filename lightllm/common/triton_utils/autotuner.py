import triton
import orjson
import os
from pathlib import Path
from tqdm import tqdm
from frozendict import frozendict
import inspect
import torch
import torch.distributed as dist
from functools import lru_cache
from lightllm.utils.device_utils import get_current_device_name
import math
import fcntl
from lightllm.utils.log_utils import init_logger
import traceback
from typing import Callable, Optional, Union, List
from lightllm.utils.envs_utils import is_triton_autotune_enabled, disable_triton_autotune
from lightllm.common.kernel_config import KernelConfigs


logger = init_logger(__name__)


def autotune(
    name: str,
    configs_gen_func: Callable[[], List],
    static_key_func: "Optional[Callable]" = None,
    run_key_func: "Optional[Callable]" = None,
    run_key_distance_func: "Optional[Callable]" = None,
    mutates_args: List[str] = None,
):
    def decorator(fn):
        arg_names = [param.name for param in inspect.signature(fn).parameters.values()]
        return Autotuner(
            fn,
            arg_names,
            name,
            configs_gen_func,
            static_key_func,
            run_key_func,
            run_key_distance_func,
            mutates_args,
        )

    return decorator


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
        configs_gen_func: Callable[[], List],
        static_key_func: Optional[Callable[[], dict]] = None,
        run_key_func: Optional[Callable] = None,
        run_key_distance_func: Optional[Callable] = lambda run_key, run_key_cached: abs(run_key - run_key_cached),
        mutates_args: List[str] = None,
    ):
        # Whether to use this autotune decorator
        self.disable_autotune = os.environ.get("DISABLE_AUTOTUNE_DECORATOR", "0") == "1"

        self.configs = None
        self.configs_gen_func = configs_gen_func

        self.name = name
        self.cache_dir = os.path.join(
            Path(__file__).parent, "all_kernel_configs", get_triton_version(), get_current_device_name(), self.name
        )
        self.fn = fn
        self.static_key_func = static_key_func
        self.run_key_func = run_key_func
        self.run_key_distance_func = run_key_distance_func
        self.cached_configs = {}
        self.arg_names = arg_names
        self._argname_to_pos = {name: idx for idx, name in enumerate(self.arg_names)}

        self._static_param_names = self._get_param_names(self.static_key_func)
        self._run_param_names = self._get_param_names(self.run_key_func)

        self.mutates_args = []
        if mutates_args is not None:
            self.mutates_args = list(mutates_args)

        self.pre_hook = lambda kwargs, reset_only=False: 0
        self.post_hook = lambda kwargs, exception: 0
        self.user_defined_pre_hook = False
        self.user_defined_post_hook = False

        if len(self.mutates_args) > 0:

            def _pre_hook(kwargs):
                self.restore_copies = {name: kwargs[name].clone() for name in self.mutates_args}

            self.pre_hook = _pre_hook

            def _post_hook(kwargs, exception):
                for name in self.restore_value:
                    kwargs[name].copy_(self.restore_copies[name])
                self.restore_copies = {}

            self.post_hook = _post_hook

        if not os.path.exists(self.cache_dir):
            if is_triton_autotune_enabled():
                os.makedirs(self.cache_dir, exist_ok=True)

        self._loaded_static_keys = set()

    @lru_cache(maxsize=None)
    def _ensure_cache_loaded(self, static_key: str):
        if static_key in self._loaded_static_keys:
            return
        cache_file = os.path.join(self.cache_dir, f"{static_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    self.cached_configs[static_key] = orjson.loads(f.read())
            except Exception:
                self.cached_configs[static_key] = {}
        self._loaded_static_keys.add(static_key)

    def _bench(self, *args, n_repeat=5, n_retries=1, **kwargs):
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

            state = BenchmarkState()
            for i in range(n_retries):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                g.replay()
                end_event.record()
                torch.cuda.synchronize()
                state.update(start_event.elapsed_time(end_event) / n_repeat)
            del g
            return state.avg
        except (OutOfResources, PTXASError, CompileTimeAssertionFailure, RuntimeError, Exception):
            return float("inf")

    def _autotune(self, args, kwargs, static_key, run_key):
        from lightllm.utils.dist_utils import get_global_rank

        if self.configs is None:
            self.configs = split_configs(self.configs_gen_func())

        rank_id = get_global_rank()
        _best_config = None
        best_time = float("inf")

        bar = tqdm(
            self.configs,
            desc=f"Autotuning {self.name} for {run_key}",
            position=get_global_rank(),
            dynamic_ncols=True,
        )
        enum_configs = enumerate(bar)
        for i, config in enum_configs:
            kwargs_with_config = kwargs.copy()
            kwargs_with_config["run_config"] = config
            run_time = self._bench(*args, **kwargs_with_config)
            if run_time < best_time:
                best_time = run_time
                _best_config = config
            bar.set_description(f"Autotuning {self.name} [rank:{rank_id}] for {run_key}, best_time: {best_time:.5f}")

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size > 1:
            local_best = torch.tensor([best_time], device="cuda")
            all_best_times = [torch.zeros_like(local_best) for _ in range(world_size)]
            dist.all_gather(all_best_times, local_best)
            all_times = [t.item() for t in all_best_times]
            min_idx = int(torch.tensor(all_times).argmin().item())
            obj_list = [_best_config]
            dist.broadcast_object_list(obj_list, src=min_idx)
            _best_config = obj_list[0]

        if static_key not in self.cached_configs:
            self.cached_configs[static_key] = {}
        self.cached_configs[static_key][run_key] = _best_config

        # save configs to file
        if not dist.is_initialized() or get_global_rank() == 0:
            cache_file = os.path.join(self.cache_dir, f"{KernelConfigs.get_config_file_name(static_key)}.json")
            with open(cache_file, "wb") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(
                        orjson.dumps(self.cached_configs[static_key], option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
                    )
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            logger.info(f"Saved configs for {self.name} - {static_key} - {run_key}")

        kwargs["run_config"] = self.cached_configs[static_key][run_key]

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        if self.disable_autotune:
            return self.fn(*args, **kwargs)

        static_key = self._static_key(*args, **kwargs)
        run_key = self._run_key(*args, **kwargs)

        # Lazy load
        self._ensure_cache_loaded(static_key)
        self.nargs = dict(zip(self.arg_names, args))

        if static_key not in self.cached_configs:
            if not is_triton_autotune_enabled():
                logger.warning(
                    f"No kernel config for {self.name} - {static_key}, \
                    using default config. Use `LIGHTLLM_TRITON_AUTOTUNE=1` to enable autotune.",
                )
            self.cached_configs[static_key] = {}

        all_configs = self.cached_configs.get(static_key)
        best_config = all_configs.get(run_key)

        if best_config is not None:
            kwargs["run_config"] = best_config
            return self.fn(*args, **kwargs)

        if is_triton_autotune_enabled():
            self._autotune(args, kwargs, static_key, run_key)
            kwargs["run_config"] = self.cached_configs.get(static_key, {}).get(run_key)
            return self.fn(*args, **kwargs)

        if all_configs != {}:
            closest_config = min(all_configs, key=lambda x: self.run_key_distance_func(int(x[0]), int(run_key)))[1]
            self.cached_configs[static_key][run_key] = closest_config
            kwargs["run_config"] = closest_config

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
        if self.static_key_func is None:
            return "default"
        params = self._select_args(self._static_param_names, args, kwargs)
        key = self.static_key_func(*params)
        return frozendict(key)

    def _run_key(self, *args, **kwargs):
        if self.run_key_func is None:
            return "default"
        params = self._select_args(self._run_param_names, args, kwargs)
        return self.run_key_func(*params)


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


def get_triton_version():
    return f"triton_{triton.__version__}"


def split_configs(configs):
    from lightllm.utils.dist_utils import get_current_rank_in_node, get_node_world_size
    import random

    random.shuffle(configs)
    rank_in_node = get_current_rank_in_node()
    node_world_size = get_node_world_size()
    return configs[rank_in_node::node_world_size]


def nearest_power_of_2(x):
    # Return the power of two closest to x
    if x <= 1:
        return 1
    return triton.next_power_of_2(x - triton.next_power_of_2(x) // 4)
