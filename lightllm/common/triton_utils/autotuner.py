import triton
import orjson
import os
import inspect
import torch
import fcntl
import torch.distributed as dist
from pathlib import Path
from tqdm import tqdm
from frozendict import frozendict
from lightllm.utils.device_utils import get_current_device_name
from lightllm.utils.log_utils import init_logger
from typing import Callable, Optional, Union, List
from lightllm.utils.envs_utils import is_triton_autotune_enabled
from lightllm.common.kernel_config import KernelConfigs


logger = init_logger(__name__)


def autotune(
    kernel_name: str,
    configs_gen_func: Callable[[], List],
    static_key_func: Callable,
    run_key_func: Callable,
    run_key_distance_func: Callable = lambda run_key, config_key: abs(int(run_key) - int(config_key)),
    mutates_args: List[str] = [],
):
    def decorator(fn):
        return Autotuner(
            fn=fn,
            kernel_name=kernel_name,
            configs_gen_func=configs_gen_func,
            static_key_func=static_key_func,
            run_key_func=run_key_func,
            run_key_distance_func=run_key_distance_func,
            mutates_args=mutates_args,
        )

    return decorator


class Autotuner:
    @staticmethod
    def _get_param_names(func: Callable) -> List[str]:
        sig = inspect.signature(func)
        return [name for name, p in sig.parameters.items()]

    def __init__(
        self,
        fn,
        kernel_name: str,
        configs_gen_func: Callable[[], List],
        static_key_func: Callable,
        run_key_func: Callable,
        run_key_distance_func: Callable = lambda run_key, config_key: abs(int(run_key) - int(config_key)),
        mutates_args: List[str] = [],
    ):
        # Whether to use this autotune decorator
        self.disable_autotune = not is_triton_autotune_enabled()

        self.configs_gen_func = configs_gen_func
        self.kernel_name = kernel_name
        self.cache_dir = os.path.join(
            Path(__file__).parent,
            "all_kernel_configs",
            get_triton_version(),
            get_current_device_name(),
            self.kernel_name,
        )
        self.fn = fn
        self.static_key_func = static_key_func
        self.run_key_func = run_key_func
        self.run_key_distance_func = run_key_distance_func
        self.cached_configs = {}
        self.arg_names = [param.name for param in inspect.signature(fn).parameters.values()]
        self._argname_to_pos = {name: idx for idx, name in enumerate(self.arg_names)}
        self._pos_to_argname = {idx: name for idx, name in enumerate(self.arg_names)}

        self._static_key_func_param_names = self._get_param_names(self.static_key_func)
        self._run_key_func_param_names = self._get_param_names(self.run_key_func)

        self.mutates_args = mutates_args

        if not os.path.exists(self.cache_dir):
            if is_triton_autotune_enabled():
                os.makedirs(self.cache_dir, exist_ok=True)

    def _try_load_cache(self, static_key):
        if static_key in self.cached_configs:
            return
        cache_file = os.path.join(self.cache_dir, KernelConfigs.get_config_file_name(static_key))
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.cached_configs[static_key] = orjson.loads(f.read())

    def _bench(self, *args, n_repeat=5, n_retries=1, **kwargs):
        from triton.compiler.errors import CompileTimeAssertionFailure
        from triton.runtime.errors import OutOfResources, PTXASError

        def kernel_call():
            new_args, new_kwargs = self._mutate_args_clone(args, kwargs)
            try:
                self.fn(*new_args, **new_kwargs)
            except Exception as e:
                raise e

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
            cache_file = os.path.join(self.cache_dir, KernelConfigs.get_config_file_name(static_key))
            with open(cache_file, "wb") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(
                        orjson.dumps(
                            self.cached_configs[static_key],
                            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS,
                        )
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
        self._try_load_cache(static_key)

        if is_triton_autotune_enabled():
            if run_key not in self.cached_configs.get(static_key, {}):
                self._autotune(args, kwargs, static_key, run_key)

        if static_key not in self.cached_configs:
            logger.warning(
                f"No kernel config for {self.kernel_name} - {static_key}, \
                using default config. Use `LIGHTLLM_TRITON_AUTOTUNE=1` to enable autotune.",
            )
            self.cached_configs[static_key] = {}

        all_configs = self.cached_configs.get(static_key)

        if len(all_configs) != 0:
            closest_config = min(all_configs, key=lambda c_key: self.run_key_distance_func(run_key, c_key))
            kwargs["run_config"] = closest_config

        return self.fn(*args, **kwargs)

    def _mutate_args_clone(self, args, kwargs):
        new_kwargs = kwargs.copy()
        new_args = list(args).copy()

        for name in self.mutates_args:
            if name in kwargs:
                new_kwargs[name] = kwargs[name].clone()
            else:
                pos = self._argname_to_pos.get(name, None)
                if pos is not None and pos < len(args):
                    new_args[pos] = args[pos].clone()
                else:
                    raise KeyError(f"Missing argument '{name}' required to be mutated")
        return tuple(new_args), new_kwargs

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
        params = self._select_args(self._static_key_func_param_names, args, kwargs)
        key = self.static_key_func(*params)
        return frozendict(key)

    def _run_key(self, *args, **kwargs):
        params = self._select_args(self._run_key_func_param_names, args, kwargs)
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
