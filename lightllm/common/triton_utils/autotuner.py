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

logger = init_logger(__name__)

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
    for k, v in sorted(data.items(), key=lambda x: str(x[0])):
        safe_k = str(k).replace(' ', '_').replace(':', '_')
        safe_v = str(v).replace(' ', '_').replace(':', '_')
        parts.append(f"{safe_k}={safe_v}")
    return ",".join(parts)


def nearest_power_of_2(x):
    # 返回最接近 x 的 2 的幂次方
    if x <= 1:
        return 1
    return triton.next_power_of_2(x - triton.next_power_of_2(x)//4) 

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
        
# Adapted from triton.testing.do_bench_cudagraph
def do_bench_cudagraph(fn, current_best_ms=None, rep=20):
    import torch

    with torch.cuda.stream(torch.cuda.Stream()):
        # warmup
        fn()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            fn()
        end_event.record()
        torch.cuda.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5
        n_repeat = max(1, int(rep / estimate_ms))
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_repeat):
                fn()
        torch.cuda.synchronize()
        # measure time and return
        state = BenchmarkState()
        n_retries = 10
        for i in range(n_retries):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            g.replay()
            end_event.record()
            torch.cuda.synchronize()
            state.update(start_event.elapsed_time(end_event) / n_repeat)
            # early stop if current kernel is bad
            if current_best_ms is not None and i >= 5:
                remaining_retries = n_retries - (i + 1)
                estimated_rem_time = remaining_retries * state.min
                if state.sum + estimated_rem_time > current_best_ms * n_retries:
                    return state.avg
            
        return state.avg


# Adapted from triton.testing.do_bench
def do_bench(fn, current_best_ms=None, warmup=25, rep=100):

    di = triton.runtime.driver.active.get_device_interface()

    fn()
    di.synchronize()

    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()

    # Estimate the runtime of the function
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        triton.runtime.driver.active.clear_cache(cache)
        fn()
    end_event.record()
    di.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = di.Event(enable_timing=True) 
    end_event = di.Event(enable_timing=True)
    
    state = BenchmarkState()

    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we clear the L2 cache before each run
        triton.runtime.driver.active.clear_cache(cache)
        # record time of `fn`
        start_event.record()
        fn()
        end_event.record()
        di.synchronize()
        use_time = start_event.elapsed_time(end_event)
        state.update(use_time)
        # early stop if current kernel is bad
        if current_best_ms is not None and i >= min(10, n_repeat // 3):
            remaining_reps = n_repeat - (i + 1)
            estimated_rem_time = remaining_reps * state.min
            if state.sum + estimated_rem_time > current_best_ms * n_repeat:
                return state.avg
    return state.avg


class Autotuner():

    @staticmethod
    def _get_param_names(func):
        if not callable(func):
            return None
        try:
            sig = inspect.signature(func)
            return [
                name for name, p in sig.parameters.items()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              inspect.Parameter.KEYWORD_ONLY)
            ]
        except (ValueError, TypeError):
            return None
        
    def __init__(self, fn, arg_names, name,configs, default_config, static_key_func, run_key_func, reset_to_zero, restore_value, pre_hook=None, post_hook=None,
                 warmup=None, rep=None, use_cuda_graph=True):
        
        self.print_autotune = os.environ.get("LIGHTLLM_TRITON_PRINT_AUTOTUNE", "0") == "1"
        self.all_configs = configs
        self.configs = None
        self.default_config = default_config
        self.name = name
        self.cache_dir = os.path.join(Path(__file__).parent, "all_kernel_configs", get_triton_version(), get_current_device_name(), self.name)
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
            _rep_full = rep if rep is not None else 50
            _rep_quick = _rep_full // 4
            self._do_bench = lambda kernel_call, current_best_ms: do_bench_cudagraph(
                kernel_call,
                rep=_rep_full,
                current_best_ms=current_best_ms,
            )
            # quick bench for staged selection
            self._do_bench_quick = lambda kernel_call, current_best_ms=None: do_bench_cudagraph(
                kernel_call,
                rep=_rep_quick,
                current_best_ms=current_best_ms,
            )
        else:
            _warmup_full = warmup if warmup is not None else 25
            _rep_full = rep if rep is not None else 100
            _warmup_quick = _warmup_full // 4
            _rep_quick = _rep_full // 4
            self._do_bench = lambda kernel_call, current_best_ms=None: do_bench(
                kernel_call,
                warmup=_warmup_full,
                rep=_rep_full,
                current_best_ms=current_best_ms,
            )
            # quick bench for staged selection
            self._do_bench_quick = lambda kernel_call, current_best_ms=None: do_bench(
                kernel_call,
                warmup=_warmup_quick,
                rep=_rep_quick,
                current_best_ms=current_best_ms,
            )

        if not os.path.exists(self.cache_dir):
            if os.environ.get("LIGHTLLM_TRITON_AUTOTUNE", "0") == "1":
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
                # 若缓存损坏，忽略并在之后覆盖
                self.cached_configs[static_key] = {}
        self._loaded_static_keys.add(static_key)
                
    def _bench(self, *args, bench="full", current_best_ms=None, **kwargs):
        from triton.compiler.errors import CompileTimeAssertionFailure
        from triton.runtime.errors import OutOfResources, PTXASError
        
        if self.print_autotune:
            logger.info(f"Autotuning kernel {self.name} with config {kwargs['run_config']}")

        full_nargs = {**self.nargs, **kwargs}

        bench_fn = self._do_bench if bench == "full" else self._do_bench_quick

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
            return bench_fn(kernel_call, current_best_ms)
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
            from lightllm.utils.dist_utils import get_global_rank
            
            rank0 = (not dist.is_initialized()) or (get_global_rank() == 0)

            # Phase 1: quick bench all configs, keep top 60%
            times_phase1 = []  # (config, time)
            current_best_ms = float('inf')
            enum_configs = enumerate(tqdm(self.configs, desc=f"Autotuning {self.fn.__name__} (phase 1:60%) for {_run_key}")) if rank0 else enumerate(self.configs)
            for i, config in enum_configs:
                kwargs_with_config = kwargs.copy()
                kwargs_with_config["run_config"] = config
                run_time = self._bench(*args, bench="quick", current_best_ms=current_best_ms, **kwargs_with_config)
                current_best_ms = min(current_best_ms, run_time)
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
                run_time = self._bench(*args, bench="quick", current_best_ms=current_best_ms, **kwargs_with_config)
                current_best_ms = min(current_best_ms, run_time)
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
            enum_configs = enumerate(tqdm(final_candidates, desc=f"Autotuning {self.name} (final) for {_run_key}")) if rank0 else enumerate(final_candidates)
            for i, config in enum_configs:
                kwargs_with_config = kwargs.copy()
                kwargs_with_config["run_config"] = config
                run_time = self._bench(*args, bench="full", current_best_ms=best_time, **kwargs_with_config)
                if run_time < best_time:
                    if rank0 and self.print_autotune and best_time != float("inf"):
                        logger.info(f"Best config for {self.name} is {_best_config} with time {best_time:.5f} -> {run_time:.5f}")
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
                if os.environ.get("LIGHTLLM_TRITON_AUTOTUNE", "0") == "1":
                    cache_file = os.path.join(self.cache_dir, f"{static_key}.json")
                    with open(cache_file, "wb") as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        try:
                            f.write(orjson.dumps(self.cached_configs[static_key], option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
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
                if static_key in self.cached_configs:
                    self.cached_configs[static_key][run_key] = self.default_config
                else:
                    logger.warning(f"No kernel config for {self.name} in {self.cache_dir}/{static_key}, using default config")
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

def autotune(name, configs, default_config, static_key_func=None, run_key_func=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None,
             use_cuda_graph=True):
    def decorator(fn):
        arg_names = [param.name for param in inspect.signature(fn).parameters.values()]
        return Autotuner(fn, arg_names, name, configs, default_config, static_key_func, run_key_func, reset_to_zero, restore_value, pre_hook=pre_hook,
                         post_hook=post_hook,use_cuda_graph=use_cuda_graph)

    return decorator

