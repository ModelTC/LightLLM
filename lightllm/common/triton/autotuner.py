 
import triton
import json
import os 
from pathlib import Path
from tqdm import tqdm
import inspect
import torch
import torch.distributed as dist
from functools import lru_cache
from lightllm.utils.device_utils import get_current_device_name
import math

@lru_cache(maxsize=1)
def get_triton_version():
    return f"triton_{triton.__version__}"


def split_configs(configs):
    from lightllm.utils.dist_utils import get_current_rank_in_node, get_node_world_size
    rank_in_node = get_current_rank_in_node()
    node_world_size = get_node_world_size()
    return configs[rank_in_node::node_world_size]


def dict_to_filename(data):
    parts = []
    # 使用确定性的键顺序，避免同一 dict 不同插入顺序导致的文件名不一致
    for k, v in sorted(data.items(), key=lambda x: str(x[0])):
        # 将键和值转为字符串并替换空格和特殊字符
        safe_k = str(k).replace(' ', '_').replace(':', '_')
        safe_v = str(v).replace(' ', '_').replace(':', '_')
        parts.append(f"{safe_k}={safe_v}")
    
    # 用下划线连接所有键值对
    return ",".join(parts)

def nearest_power_of_2(x):
    if x <= 1:
        return 1
    return triton.next_power_of_2(x - triton.next_power_of_2(x)//4) 

class Autotuner():

    def __init__(self, fn, arg_names, configs, default_config, static_key_func, run_key_func, reset_to_zero, restore_value, pre_hook=None, post_hook=None,
                 warmup=None, rep=None, use_cuda_graph=False):
        
        self.enable_autotune = os.environ.get("LIGHTLLM_ENABLE_AUTOTUNE", "0") == "1"
        self.print_autotune = os.environ.get("LIGHTLLM_PRINT_AUTOTUNE", "0") == "1"
        self.all_configs = configs
        self.configs = None
        self.default_config = default_config
        self.unique_id = f"{fn.__module__}.{fn.__name__}" 
        self.cache_dir = os.path.join(Path(__file__).parent, "all_kernel_configs", get_triton_version(), get_current_device_name(), self.unique_id)
        self.fn = fn
        self.static_key_func = static_key_func
        self.run_key_func = run_key_func
        
        self.cached_configs = {}
        self.arg_names = arg_names
        self._argname_to_pos = {name: idx for idx, name in enumerate(self.arg_names)}

        # Precompute signatures for fast argument selection
        self._static_param_names = None
        self._run_param_names = None
        if callable(self.static_key_func):
            try:
                sig = inspect.signature(self.static_key_func)
                self._static_param_names = [
                    name for name, p in sig.parameters.items()
                    if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                  inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                  inspect.Parameter.KEYWORD_ONLY)
                ]
            except (ValueError, TypeError):
                self._static_param_names = None
        if callable(self.run_key_func):
            try:
                sig = inspect.signature(self.run_key_func)
                self._run_param_names = [
                    name for name, p in sig.parameters.items()
                    if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                  inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                  inspect.Parameter.KEYWORD_ONLY)
                ]
            except (ValueError, TypeError):
                self._run_param_names = None

        # Reset to zero or restore values
        self.reset_to_zero = []
        if reset_to_zero is not None:
            self.reset_to_zero = list(reset_to_zero)
        self.restore_value = []
        if restore_value is not None:
            self.restore_value = list(restore_value)

        # Hook to reset or restore for required tensors
        self.pre_hook = lambda kwargs, reset_only=False: 0
        self.post_hook = lambda kwargs, exception: 0
        self.user_defined_pre_hook = False
        self.user_defined_post_hook = False
        if pre_hook:
            self.pre_hook = pre_hook
            self.user_defined_pre_hook = True
        elif (len(self.reset_to_zero) > 0 or len(self.restore_value) > 0):

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

        
        self.use_cuda_graph = use_cuda_graph
        
        if use_cuda_graph:
            self._do_bench = lambda kernel_call, quantiles: triton.testing.do_bench_cudagraph(
                kernel_call,
                rep=rep if rep is not None else 100,
                quantiles=quantiles,
            )
            # quick bench for staged selection
            self._do_bench_quick = lambda kernel_call, quantiles: triton.testing.do_bench_cudagraph(
                kernel_call,
                rep=20,
                quantiles=quantiles,
            )
        else:
            self._do_bench = lambda kernel_call, quantiles: triton.testing.do_bench(
                kernel_call,
                warmup=warmup if warmup is not None else 25,
                rep=rep if rep is not None else 100,
                quantiles=quantiles,
            )
            # quick bench for staged selection
            self._do_bench_quick = lambda kernel_call, quantiles: triton.testing.do_bench(
                kernel_call,
                warmup=5,
                rep=20,
                quantiles=quantiles,
            )

        if not os.path.exists(self.cache_dir):
            if self.enable_autotune:
                os.makedirs(self.cache_dir, exist_ok=True)

        self._loaded_static_keys = set()

    @lru_cache(maxsize=None)
    def _ensure_cache_loaded(self, static_key: str):
        if static_key in self._loaded_static_keys:
            return
        cache_file = os.path.join(self.cache_dir, f"{static_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    self.cached_configs[static_key] = json.load(f)
            except Exception:
                # 若缓存损坏，忽略并在之后覆盖
                self.cached_configs[static_key] = {}
        self._loaded_static_keys.add(static_key)
                
    def _bench(self, *args, bench="full", **kwargs):
        from triton.compiler.errors import CompileTimeAssertionFailure
        from triton.runtime.errors import OutOfResources, PTXASError
        
        if self.print_autotune:
            print(f"Autotuning kernel {self.fn.__name__} with config {kwargs['run_config']}")

        full_nargs = {**self.nargs, **kwargs}

        can_use_fast_reset = (not self.user_defined_pre_hook) and (not self.user_defined_post_hook)
        bench_fn = self._do_bench if bench == "full" else self._do_bench_quick

        if can_use_fast_reset and (len(self.reset_to_zero) > 0 or len(self.restore_value) > 0):
            restore_copies = {name: full_nargs[name].clone() for name in self.restore_value}

            def fast_kernel_call():
                for name in self.restore_value:
                    full_nargs[name].copy_(restore_copies[name])
                for name in self.reset_to_zero:
                    full_nargs[name].zero_()
                self.fn(*args, **kwargs)

            try:
                result = bench_fn(fast_kernel_call, quantiles=(0.5,))
                for name in self.restore_value:
                    full_nargs[name].copy_(restore_copies[name])
                return result
            except (OutOfResources, PTXASError, CompileTimeAssertionFailure, RuntimeError, Exception):
                return float("inf")
        else:
            def kernel_call():
                self.pre_hook(full_nargs)
                try:
                    self.fn(*args, **kwargs)
                except Exception as e:
                    try:
                        self.post_hook(full_nargs, exception=e)
                    finally:
                        # Throw exception raised by `self.fn.run`
                        raise

                self.post_hook(full_nargs, exception=None)

        try:
            return bench_fn(kernel_call, quantiles=(0.5,))  
        except (OutOfResources, PTXASError, CompileTimeAssertionFailure, RuntimeError, Exception):
            return float("inf")

    def __call__(self, *args, **kwargs):
        if self.configs is None:
            self.configs = split_configs(self.all_configs)

        static_key = self._static_key(*args, **kwargs)
        run_key = self._run_key(*args, **kwargs)
        
        # 懒加载
        self._ensure_cache_loaded(static_key)
        best_config = None
        self.nargs = dict(zip(self.arg_names, args))
        def _benchmark(_run_key):
            from lightllm.utils.dist_utils import get_global_rank, get_current_rank_in_node
            torch.cuda.set_device(get_current_rank_in_node())
            
            rank0 = (not dist.is_initialized()) or (get_global_rank() == 0)

            # Phase 1: quick bench all configs, keep top 60%
            times_phase1 = []  # (config, time)
            enum_configs = enumerate(tqdm(self.configs, desc=f"Autotuning {self.fn.__name__} (phase 1:60%) for {_run_key}")) if rank0 else enumerate(self.configs)
            for i, config in enum_configs:
                kwargs_with_config = kwargs.copy()
                kwargs_with_config["run_config"] = config
                run_time = self._bench(*args, bench="quick", **kwargs_with_config)
                times_phase1.append((config, run_time))

            times_phase1.sort(key=lambda x: x[1])
            k60 = max(1, int(math.ceil(len(times_phase1) * 0.6)))
            top60 = [cfg for cfg, _t in times_phase1[:k60]]

            # Phase 2: quick bench top60, keep top 30%
            times_phase2 = []
            enum_configs = enumerate(tqdm(top60, desc=f"Autotuning {self.fn.__name__} (phase 2:30%) for {_run_key}")) if rank0 else enumerate(top60)
            for i, config in enum_configs:
                kwargs_with_config = kwargs.copy()
                kwargs_with_config["run_config"] = config
                run_time = self._bench(*args, bench="quick", **kwargs_with_config)
                times_phase2.append((config, run_time))

            times_phase2.sort(key=lambda x: x[1])
            k30 = max(1, int(math.ceil(len(times_phase2) * 0.5)))
            top30 = [cfg for cfg, _t in times_phase2[:k30]]

            # Phase 3: full bench final candidates (+ default)
            final_candidates = list(top30)
            if self.default_config not in final_candidates:
                final_candidates.append(self.default_config)

            _best_config = self.default_config
            best_time = float("inf")
            enum_configs = enumerate(tqdm(final_candidates, desc=f"Autotuning {self.fn.__name__} (final) for {_run_key}")) if rank0 else enumerate(final_candidates)
            for i, config in enum_configs:
                kwargs_with_config = kwargs.copy()
                kwargs_with_config["run_config"] = config
                run_time = self._bench(*args, bench="full", **kwargs_with_config)
                if run_time < best_time:
                    if rank0 and self.print_autotune and best_time != float("inf"):
                        print(f"Best config for {self.fn.__name__} is {_best_config} with time {best_time:.5f} -> {run_time:.5f}")
                    best_time = run_time
                    _best_config = config
                    
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
            
            if not dist.is_initialized() or get_global_rank() == 0:
                if self.enable_autotune:
                    cache_file = os.path.join(self.cache_dir, f"{static_key}.json")
                    with open(cache_file, "w") as f:
                        json.dump(self.cached_configs[static_key], f, indent=2, sort_keys=True)
                    if self.print_autotune:
                        print(f"Cached configs: {self.cached_configs[static_key]}")
            
            kwargs["run_config"] = self.cached_configs[static_key][run_key]
            full_nargs = {**self.nargs, **kwargs}
            self.pre_hook(full_nargs, reset_only=True)
            
        if static_key in self.cached_configs:
            if run_key in self.cached_configs[static_key]:
                best_config = self.cached_configs[static_key][run_key]

        if best_config is None and self.enable_autotune:
            _benchmark(run_key)
            best_config = self.cached_configs[static_key][run_key]
            
        kwargs["run_config"] = best_config if best_config is not None else self.default_config
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
                key = self.static_key_func(*params) if self._static_param_names is not None else self.static_key_func(*args, **kwargs)
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
                key = self.run_key_func(*params) if self._run_param_names is not None else self.run_key_func(*args, **kwargs)
            except Exception:
                key = self.run_key_func(*args, **kwargs)
            if isinstance(key, dict):
                return dict_to_filename(key)
            return str(key)
        return "default"

def autotune(configs, default_config, static_key_func=None, run_key_func=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None,
             warmup=None, rep=None, use_cuda_graph=False):
    def decorator(fn):
        arg_names = [param.name for param in inspect.signature(fn).parameters.values()]
        return Autotuner(fn, arg_names, configs, default_config, static_key_func, run_key_func, reset_to_zero, restore_value, pre_hook=pre_hook,
                         post_hook=post_hook, warmup=warmup, rep=rep,
                         use_cuda_graph=use_cuda_graph)

    return decorator

