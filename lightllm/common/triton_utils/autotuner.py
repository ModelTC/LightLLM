import triton
import orjson
import os
import inspect
import torch
import torch.distributed as dist
import random
import collections
from pathlib import Path
from tqdm import tqdm
from frozendict import frozendict
from lightllm.utils.device_utils import get_current_device_name
from lightllm.utils.log_utils import init_logger
from typing import Callable, Optional, Union, List
from lightllm.utils.envs_utils import get_triton_autotune_level
from lightllm.common.kernel_config import KernelConfigs
from lightllm.utils.dist_utils import get_global_world_size, get_global_rank, get_current_rank_in_node

logger = init_logger(__name__)


class AutotuneLevel:
    # Use the config of cached files in /lightllm/common/triton_utils/autotune_kernel_configs.
    USE_AUTOTUNE_HIS_CONFIG = 0
    # Autotune if no config is cached.
    ADAPTIVE_AUTOTUNE = 1
    # Autotune anyway to overwrite the config of cached files.
    FORCE_AUTOTUNE = 2
    # Close autotune and use the configs of cached files in lightllm/common/all_kernel_configs.
    CLOSE_AUTOTUNE = 3


def autotune(
    kernel_name: str,
    configs_gen_func: Callable[[], List],
    static_key_func: Callable,
    run_key_func: Callable,
    run_key_distance_func: Callable = lambda run_key, config_key: abs(int(run_key) - int(config_key)),
    mutates_args: List[str] = [],
):
    """Decorator that constructs and returns an Autotuner wrapper for a Triton kernel.

    This decorator configures an Autotuner with the provided configuration
    generator and key functions, enabling on-demand benchmarking and caching
    of kernel run configurations across runs and processes.

    Args:
        kernel_name (str): Human-readable kernel name used for logging and cache paths.
        configs_gen_func (Callable[[], List]): Function that returns candidate run configurations.
        static_key_func (Callable): Function that derives a static key (dict-like) from call arguments.
            This key identifies the cache file that stores tuned configs.
        run_key_func (Callable): Function that derives a run-time key from call arguments.
            This key indexes tuned configs within a static key's cache.
        run_key_distance_func (Callable, optional): Distance metric taking ``(run_key, config_key)`` and
            returning a comparable value; used to pick the closest config when an exact match is absent.
            Defaults to ``abs(int(run_key) - int(config_key))``.
        mutates_args (List[str], optional): Names of arguments that can be mutated by the kernel.
            During benchmarking, defensive clones are made to avoid side effects. Defaults to ``[]``.

    Returns:
        Callable: A callable object that wraps the original function and performs autotuning
        as needed before invocation.
    """

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
    _autotune_warmup: bool = False

    @staticmethod
    def start_autotune_warmup():
        Autotuner._autotune_warmup = True
        return

    @staticmethod
    def end_autotune_warmup():
        Autotuner._autotune_warmup = False
        return

    @staticmethod
    def is_autotune_warmup():
        return Autotuner._autotune_warmup

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

        self.configs_gen_func = configs_gen_func
        self.kernel_name = kernel_name
        self.cache_dir = os.path.join(
            Path(__file__).parent,
            "autotune_kernel_configs",
            get_triton_version(),
            get_current_device_name(),
            self.kernel_name,
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        self.fn = fn
        self.static_key_func = static_key_func
        self.run_key_func = run_key_func
        self.run_key_distance_func = run_key_distance_func
        self.cached_configs = {}
        self.fast_match_configs = collections.defaultdict(dict)
        self.warmuped_configs_set = set()
        self.arg_names = [param.name for param in inspect.signature(self.fn).parameters.values()]
        self._argname_to_pos = {name: idx for idx, name in enumerate(self.arg_names)}
        self._pos_to_argname = {idx: name for idx, name in enumerate(self.arg_names)}

        self._static_key_func_param_names = [
            name for name, _ in inspect.signature(self.static_key_func).parameters.items()
        ]
        self._run_key_func_param_names = [name for name, _ in inspect.signature(self.run_key_func).parameters.items()]
        self.mutates_args = mutates_args

        assert get_triton_autotune_level() in [
            AutotuneLevel.USE_AUTOTUNE_HIS_CONFIG,
            AutotuneLevel.ADAPTIVE_AUTOTUNE,
            AutotuneLevel.FORCE_AUTOTUNE,
            AutotuneLevel.CLOSE_AUTOTUNE,
        ]
        return

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        if kwargs.get("run_config", None) is not None:
            return self.fn(*args, **kwargs)

        # if the autotune_level is AutotuneLevel.CLOSE_AUTOTUNE, ignore the autotune
        autotune_level = get_triton_autotune_level()
        if autotune_level == AutotuneLevel.CLOSE_AUTOTUNE:
            return self.fn(*args, **kwargs)

        rank_id = 0 if not dist.is_initialized() else get_global_rank()
        world_size = 1 if not dist.is_initialized() else get_global_world_size()

        static_key = frozendict(self._static_key(*args, **kwargs))
        run_key = str(self._run_key(*args, **kwargs))

        # Lazy load the cached configs in lightllm/common/triton_utils/autotune_kernel_configs
        if self._try_load_cache(static_key) or Autotuner.is_autotune_warmup():
            all_configs = self.cached_configs.get(static_key, {})
            for run_config in all_configs.values():
                # warmup all configs
                _copy_kwargs = kwargs.copy()
                _copy_kwargs["run_config"] = run_config
                self.kernel_warmup(static_key, *args, **_copy_kwargs)

        if static_key not in self.cached_configs and autotune_level == AutotuneLevel.USE_AUTOTUNE_HIS_CONFIG:
            if (dist.is_initialized() and get_current_rank_in_node() == 0) or not dist.is_initialized():
                logger.warning(
                    f"No kernel config for {self.kernel_name} in {KernelConfigs.get_config_file_name(static_key)},"
                    f"the performance may be suboptimal!"
                    f"You can use LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 to enable autotune.",
                )
            self.cached_configs[static_key] = {}

        if (
            autotune_level in [AutotuneLevel.ADAPTIVE_AUTOTUNE, AutotuneLevel.FORCE_AUTOTUNE]
            and Autotuner.is_autotune_warmup()
        ):
            need_tuning = (autotune_level == AutotuneLevel.FORCE_AUTOTUNE) or (
                run_key not in self.cached_configs.get(static_key, {})
            )
            if world_size > 1:
                _need_tunings = [None for _ in range(world_size)]
                dist.all_gather_object(_need_tunings, obj=need_tuning, group=self._get_autotune_group())
                need_tuning = any(_need_tunings)
            if need_tuning:
                self._autotune(
                    args=args,
                    kwargs=kwargs,
                    static_key=static_key,
                    run_key=run_key,
                    rank_id=rank_id,
                    world_size=world_size,
                )

        closest_config = self.fast_match_configs.get(static_key, {}).get(run_key, None)
        if closest_config is not None:
            kwargs["run_config"] = closest_config
            return self.fn(*args, **kwargs)

        all_configs = self.cached_configs.get(static_key, {})
        if len(all_configs) != 0:
            closest_config = min(
                list(all_configs.items()), key=lambda item: self.run_key_distance_func(run_key, item[0])
            )[1]
            kwargs["run_config"] = closest_config
            self.fast_match_configs[static_key][run_key] = closest_config

        return self.fn(*args, **kwargs)

    def _try_load_cache(self, static_key):
        if static_key in self.cached_configs:
            return False

        cache_file = os.path.join(self.cache_dir, KernelConfigs.get_config_file_name(static_key))
        if os.path.exists(cache_file):
            logger.info(f"Loading cached configs for {self.kernel_name} - {static_key}")
            with open(cache_file, "rb") as f:
                self.cached_configs[static_key] = orjson.loads(f.read())
        return True

    def kernel_warmup(self, static_key, *args, **kwargs):
        new_args, new_kwargs, origin_list, new_list = self._mutate_args_clone(args, kwargs)
        run_config = kwargs.get("run_config", {})
        hash_key = str(frozendict(run_config)) + str(static_key)
        if hash_key in self.warmuped_configs_set:
            return
        try:
            self.fn(*new_args, **new_kwargs)
            self.warmuped_configs_set.add(hash_key)
        except:
            pass
        finally:
            self._recover_mutated_args(origin_list=origin_list, new_list=new_list)
        return

    def _bench(self, *args, n_repeat=3, n_retries=3, **kwargs):
        from triton.compiler.errors import CompileTimeAssertionFailure
        from triton.runtime.errors import OutOfResources, PTXASError

        new_args, new_kwargs, origin_list, new_list = self._mutate_args_clone(args, kwargs)

        def kernel_call():
            try:
                self.fn(*new_args, **new_kwargs)
            except Exception as e:
                raise e
            finally:
                self._recover_mutated_args(origin_list=origin_list, new_list=new_list)

        try:
            # warmup
            kernel_call()

            torch.cuda.current_stream().synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=torch.cuda.Stream()):
                for _ in range(n_repeat):
                    kernel_call()
            torch.cuda.current_stream().synchronize()

            state = _BenchmarkState()
            for i in range(n_retries):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                g.replay()
                end_event.record()
                end_event.synchronize()
                state.update(start_event.elapsed_time(end_event) / n_repeat)
            del g
            return state.avg
        except (OutOfResources, PTXASError, CompileTimeAssertionFailure, RuntimeError, Exception):
            return float("inf")

    def _autotune(self, args, kwargs, static_key, run_key, rank_id, world_size):
        is_key_all_same = True
        if world_size > 1:
            all_keys = [None for _ in range(world_size)]
            all_key_str = f"{run_key}_{static_key}"
            dist.all_gather_object(all_keys, obj=all_key_str, group=self._get_autotune_group())
            is_key_all_same = all(all_keys[0] == k for k in all_keys)
            if not is_key_all_same:
                logger.warning(
                    f"{self.kernel_name} not all key is same, get keys {all_keys}, tuning is not parral split configs"
                )
                rank_tuning_configs = self.configs_gen_func()
            else:
                rank_tuning_configs = split_configs(
                    self.configs_gen_func(), global_rank=rank_id, global_world_size=world_size
                )
        else:
            rank_tuning_configs = self.configs_gen_func()

        best_config = None
        best_time = float("inf")

        bar = tqdm(
            rank_tuning_configs,
            desc=f"Autotuning {self.kernel_name} for {run_key}",
            position=rank_id,
            dynamic_ncols=True,
        )
        enum_configs = enumerate(bar)
        for i, config in enum_configs:
            kwargs_with_config = kwargs.copy()
            kwargs_with_config["run_config"] = config
            run_time = self._bench(*args, **kwargs_with_config)
            if run_time < best_time:
                best_time = run_time
                best_config = config
            bar.set_description(
                f"Autotuning {self.kernel_name} [rank:{rank_id}] for {run_key}, best_time: {best_time:.5f}"
            )

        update_static_key_list = []
        if world_size > 1:
            all_gather_configs = [None for _ in range(world_size)]
            dist.all_gather_object(
                all_gather_configs,
                obj=(best_time, run_key, dict(static_key), best_config),
                group=self._get_autotune_group(),
            )
            all_gather_configs = sorted(all_gather_configs, key=lambda x: x[0])
            key_set = set()
            unique_configs = collections.defaultdict(dict)
            for _best_time, _run_key, _static_key, _config in all_gather_configs:
                _all_key = f"{_run_key}_{frozendict(_static_key)}"
                update_static_key_list.append(frozendict(_static_key))
                if _all_key not in key_set:
                    unique_configs[frozendict(_static_key)][_run_key] = _config
                    key_set.add(_all_key)
        else:
            unique_configs = collections.defaultdict(dict)
            unique_configs[static_key][run_key] = best_config
            update_static_key_list.append(static_key)

        for _static_key, _t_dict in unique_configs.items():
            if _static_key not in self.cached_configs:
                self.cached_configs[_static_key] = {}
            for _run_key, _config in _t_dict.items():
                self.cached_configs[_static_key][_run_key] = _config

        # save configs to file
        if rank_id == 0:
            for _static_key in update_static_key_list:
                cache_file = os.path.join(self.cache_dir, KernelConfigs.get_config_file_name(_static_key))
                with open(cache_file, "wb") as f:
                    f.write(
                        orjson.dumps(
                            self.cached_configs[_static_key],
                            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS,
                        )
                    )
                logger.info(f"Saved configs for {self.kernel_name} - {_static_key}")

        logger.info(f"rank {rank_id} tuning {self.kernel_name} _static_key {static_key} finished")

    def _mutate_args_clone(self, args, kwargs):
        origin_list = []
        new_list = []
        new_kwargs = kwargs.copy()
        new_args = list(args).copy()

        for name in self.mutates_args:
            if name in kwargs:
                new_kwargs[name] = None if kwargs[name] is None else kwargs[name].clone()
                origin_list.append(kwargs[name])
                new_list.append(new_kwargs[name])
            else:
                pos = self._argname_to_pos.get(name, None)
                if pos is not None and pos < len(args):
                    new_args[pos] = None if args[pos] is None else args[pos].clone()
                    origin_list.append(args[pos])
                    new_list.append(new_args[pos])
                else:
                    raise KeyError(f"Missing argument '{name}' required to be mutated")
        return tuple(new_args), new_kwargs, origin_list, new_list

    def _recover_mutated_args(self, origin_list, new_list):
        for a, b in zip(origin_list, new_list):
            if b is not None:
                b.copy_(a)
        return

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
        return self.static_key_func(*params)

    def _run_key(self, *args, **kwargs):
        params = self._select_args(self._run_key_func_param_names, args, kwargs)
        return self.run_key_func(*params)

    def _get_autotune_group(
        self,
    ):
        from lightllm.distributed.communication_op import dist_group_manager

        return dist_group_manager.get_default_group().autotune_group


class _BenchmarkState:
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


def split_configs(configs, global_rank, global_world_size):
    random.Random(0).shuffle(configs)
    return configs[global_rank::global_world_size]
