from typing import Dict, Tuple, List, Optional
import triton
import json
import os 
from pathlib import Path
import functools
import time
import builtins
from tqdm import tqdm
import inspect
import torch
import torch.distributed as dist

from lightllm.utils.device_utils import get_current_device_name
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

def get_triton_version():
    return f"triton_{triton.__version__}"


def split_configs(configs):
    from lightllm.utils.dist_utils import get_current_rank_in_node, get_node_world_size
    rank_in_node = get_current_rank_in_node()
    node_world_size = get_node_world_size()
    return configs[rank_in_node::node_world_size]


class Autotuner():

    def __init__(self, fn, arg_names, configs, default_config, static_key_func, run_key_func, reset_to_zero, restore_value, pre_hook=None, post_hook=None,
                 prune_configs_by: Optional[Dict] = None, warmup=None, rep=None, use_cuda_graph=False):
        
        self.enable_autotune = os.environ.get("LIGHTLLM_ENABLE_AUTOTUNE", "0") == "1"
        self.print_autotune = os.environ.get("LIGHTLLM_PRINT_AUTOTUNE", "0") == "1"
        
        self.configs = split_configs(configs)
        self.default_config = default_config
        self.unique_id = f"{fn.__module__}.{fn.__name__}" 
        self.cache_dir = os.path.join(Path(__file__).parent, "all_kernel_configs", get_triton_version(), get_current_device_name(), self.unique_id)
        self.fn = fn
        self.static_key_func = static_key_func
        self.run_key_func = run_key_func
        
        self.cached_configs = {}
        self.arg_names = arg_names

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

        self.num_warmups = warmup
        self.num_reps = rep
        self.use_cuda_graph = use_cuda_graph
        
        if use_cuda_graph:
            self._do_bench = lambda kernel_call, quantiles: triton.testing.do_bench_cudagraph(
                kernel_call,
                rep=rep if rep is not None else 100,
                quantiles=quantiles,
            )
        else:
            self._do_bench = lambda kernel_call, quantiles: triton.testing.do_bench(
                kernel_call,
                warmup=warmup if warmup is not None else 25,
                rep=rep if rep is not None else 100,
                quantiles=quantiles,
            )

        if os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                if file.endswith(".json"):
                    with open(os.path.join(self.cache_dir, file), "r") as f:
                        self.cached_configs[file.split(".")[0]] = json.load(f)
        else:
            if self.enable_autotune:
                os.makedirs(self.cache_dir, exist_ok=True)
        
    def _bench(self, *args, **kwargs):
        from triton.compiler.errors import CompileTimeAssertionFailure
        from triton.runtime.errors import OutOfResources, PTXASError
        
        if self.print_autotune:
            print(f"Autotuning kernel {self.fn.__name__} with config {kwargs['run_config']}")

        full_nargs = {**self.nargs, **kwargs}

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
            return self._do_bench(kernel_call, quantiles=(0.5,))  
        except (OutOfResources, CompileTimeAssertionFailure, PTXASError) as e:
            if self.print_autotune:
                print(f"Autotuning failed with {e}")
            return float("inf")

    def __call__(self, *args, **kwargs):
                    
        static_key = self.static_key_func(*args, **kwargs)
        run_key = self.run_key_func(*args, **kwargs)
        best_config = None
        self.nargs = dict(zip(self.arg_names, args))
        def _benchmark(_run_key):
            from lightllm.utils.dist_utils import get_global_rank
            
            patience = len(self.configs) // 3
            best_config = None
            best_time = float("inf")
            enum_configs = enumerate(tqdm(self.configs, desc=f"Autotuning {self.fn.__name__}::{_run_key}")) if not dist.is_initialized() or get_global_rank() == 0 else enumerate(self.configs)
            for i, config in enum_configs:
                kwargs_with_config = kwargs.copy()
                kwargs_with_config["run_config"] = config
                run_time = self._bench(*args, **kwargs_with_config)
                if run_time < best_time:
                    best_time = run_time
                    best_config = config
                    if not dist.is_initialized() or get_global_rank() == 0:
                        print(f"Best config for {self.fn.__name__} is {best_config} with time {best_time}")
                    patience = len(self.configs) // 3
                else:
                    patience -= 1
                    if patience <= 0:
                        break

            # 收集所有进程的best_time和best_config
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            local_best = torch.tensor([best_time], device="cuda" if torch.cuda.is_available() else "cpu")
            all_best_times = [torch.zeros_like(local_best) for _ in range(world_size)]
            if dist.is_initialized():
                dist.all_gather(all_best_times, local_best)
            else:
                all_best_times = [local_best]

            # 收集所有进程的best_config
            import pickle
            local_config_bytes = pickle.dumps(best_config)
            config_length = len(local_config_bytes)
            
            # 首先收集配置长度
            local_length = torch.tensor([config_length], dtype=torch.long, device=local_best.device)
            all_lengths = [torch.zeros_like(local_length) for _ in range(world_size)]
            if dist.is_initialized():
                dist.all_gather(all_lengths, local_length)
            else:
                all_lengths = [local_length]
            
            # 找到最大长度
            max_length = max(length.item() for length in all_lengths)
            
            # 创建足够大的tensor来存储配置
            local_config_tensor = torch.zeros(max_length, dtype=torch.uint8, device=local_best.device)
            config_bytes = torch.tensor(list(local_config_bytes), dtype=torch.uint8, device=local_best.device)
            local_config_tensor[:config_length] = config_bytes
            all_config_tensors = [torch.zeros_like(local_config_tensor) for _ in range(world_size)]
            if dist.is_initialized():
                dist.all_gather(all_config_tensors, local_config_tensor)
            else:
                all_config_tensors = [local_config_tensor]

            # 找到最小时间的进程
            all_times = [t.item() for t in all_best_times]
            min_idx = int(torch.tensor(all_times).argmin().item())

            # 选用对应的best_config
            min_length = all_lengths[min_idx].item()
            best_config_bytes = bytes([int(x) for x in all_config_tensors[min_idx].cpu().numpy()[:min_length]])
            best_config = pickle.loads(best_config_bytes)
            
            if static_key not in self.cached_configs:
                self.cached_configs[static_key] = {}
            self.cached_configs[static_key][run_key] = best_config
            
            # 只在rank 0进行持久化
            if not dist.is_initialized() or get_global_rank() == 0:
                if self.enable_autotune:
                    cache_file = os.path.join(self.cache_dir, f"{static_key}.json")
                    with open(cache_file, "w") as f:
                        json.dump(self.cached_configs[static_key], f, indent=2)
            
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
        
    
def autotune(configs, default_config, static_key_func, run_key_func, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None,
             warmup=None, rep=None, use_cuda_graph=False):
    def decorator(fn):
        arg_names = [param.name for param in inspect.signature(fn).parameters.values()]
        return Autotuner(fn, arg_names, configs, default_config, static_key_func, run_key_func, reset_to_zero, restore_value, pre_hook=pre_hook,
                         post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                         use_cuda_graph=use_cuda_graph)

    return decorator


def nearest_power_of_2(x):
    if x <= 1:
        return 1
    return triton.next_power_of_2(x - triton.next_power_of_2(x)//4) 