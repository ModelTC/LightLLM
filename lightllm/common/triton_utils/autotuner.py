import triton
import orjson
import os
import inspect
import torch
import torch.distributed as dist
import random
import collections
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from tqdm import tqdm
from frozendict import frozendict
from lightllm.utils.device_utils import get_current_device_name
from lightllm.utils.log_utils import init_logger
from typing import Callable, List, Optional
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


class AutotuneKernelType(str, Enum):
    GENERAL = "general"
    # Includes full-attention and linear-attention decode kernels.
    DECODE_ATTENTION = "decode_attention"


def autotune(
    kernel_name: str,
    configs_gen_func: Callable[[], List],
    static_key_func: Callable,
    run_key_func: Callable,
    run_key_distance_func: Callable = lambda run_key, config_key: abs(int(run_key) - int(config_key)),
    mutates_args: List[str] = [],
    kernel_type: AutotuneKernelType = AutotuneKernelType.GENERAL,
    rebuild_input_func: Optional[Callable] = None,
    warmup_all_exist_config: bool = True,
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
        kernel_type (AutotuneKernelType, optional): Only a matching warmup phase benchmarks this kernel.
            Other phases still execute it using cached configurations or its default configuration.
        rebuild_input_func (Callable, optional): 调优输入重建回调，主要供 decode attention 算子使用。
            CUDA Graph 初始化时，输入的真实请求长度通常很短，无法代表实际 decode 场景的计算量，
            因此需要算子通过此回调自行重建输入，例如填入目标 KV 长度并构造对应的合法页表。
            每次实际调优搜索前调用一次，接收算子的原始参数，返回用于计时的 ``(args, kwargs)``。
            回调不应修改原始输入；缓存键、历史配置预热及最终执行仍使用原始参数。
        warmup_all_exist_config (bool, optional): 是否提前执行所有已有配置进行预热，默认 True。
            设为 False 后，首次加载缓存和任何 warmup 阶段都不执行这一步，但仍正常加载、选择配置。
            原地更新持久状态且无法低成本保存/恢复的算子应关闭，例如 MTP linear attention 的
            SSM 递推、原地追加 KV 或累加持久统计量的算子：每次预热都会额外推进或重复写入状态，
            可能改变后续正式计算的结果；把大型状态池加入 mutates_args 又会因 clone 增加显存占用，
            甚至触发 OOM。仅覆盖输出缓冲区，或可通过 mutates_args 完整保护输入的算子可保持默认值。
            当前关闭该开关的特殊算子是 ``mtp_fused_recurrent_gated_delta_rule``，对应 autotune
            ``kernel_name`` 为 ``_mtp_fused_recurrent_gated_delta_rule_fwd_kernel:v1``，可用这两个名字
            查询实现、调用位置和缓存配置目录。
            此开关只控制已有配置的额外预热，不关闭新配置的搜索、benchmark 内部的预热/计时和
            最终正式执行；实际搜索仍需由调用方保证状态可以被反复更新，或提供相应的状态保护。

    Returns:
        Callable: A callable object that wraps the original function and performs autotuning
        as needed before invocation.
    """

    def decorator(fn: Callable) -> Callable:
        return Autotuner(
            fn=fn,
            kernel_name=kernel_name,
            configs_gen_func=configs_gen_func,
            static_key_func=static_key_func,
            run_key_func=run_key_func,
            run_key_distance_func=run_key_distance_func,
            mutates_args=mutates_args,
            kernel_type=kernel_type,
            rebuild_input_func=rebuild_input_func,
            warmup_all_exist_config=warmup_all_exist_config,
        )

    return decorator


class Autotuner:
    _autotune_warmup_kernel_type: Optional[AutotuneKernelType] = None

    @staticmethod
    def start_autotune_warmup(kernel_type: AutotuneKernelType = AutotuneKernelType.GENERAL):
        """Select the kernel category to tune; all distributed ranks must select the same phase."""
        Autotuner._autotune_warmup_kernel_type = AutotuneKernelType(kernel_type)
        return

    @staticmethod
    def end_autotune_warmup():
        Autotuner._autotune_warmup_kernel_type = None
        return

    @staticmethod
    def is_autotune_warmup() -> bool:
        """Report whether any warmup phase is active."""
        return Autotuner._autotune_warmup_kernel_type is not None

    @staticmethod
    def is_kernel_autotune_warmup(kernel_type: AutotuneKernelType) -> bool:
        """Report whether this kernel category is selected for warmup."""
        return Autotuner._autotune_warmup_kernel_type == AutotuneKernelType(kernel_type)

    @staticmethod
    @contextmanager
    def autotune_warmup(kernel_type: AutotuneKernelType = AutotuneKernelType.GENERAL):
        """Restore the previous warmup phase on exit, including nested scopes and exceptions."""
        previous_type = Autotuner._autotune_warmup_kernel_type
        Autotuner.start_autotune_warmup(kernel_type)
        try:
            yield
        finally:
            Autotuner._autotune_warmup_kernel_type = previous_type

    def __init__(
        self,
        fn,
        kernel_name: str,
        configs_gen_func: Callable[[], List],
        static_key_func: Callable,
        run_key_func: Callable,
        run_key_distance_func: Callable = lambda run_key, config_key: abs(int(run_key) - int(config_key)),
        mutates_args: List[str] = [],
        kernel_type: AutotuneKernelType = AutotuneKernelType.GENERAL,
        rebuild_input_func: Optional[Callable] = None,
        warmup_all_exist_config: bool = True,
    ):

        self.configs_gen_func = configs_gen_func
        self.kernel_name = kernel_name
        self.kernel_type = AutotuneKernelType(kernel_type)
        self.rebuild_input_func = rebuild_input_func
        self.warmup_all_exist_config = warmup_all_exist_config
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

        # decode attention 在多层中会重复调用，相同配置只需调优一次，避免强制调优拖慢启动。
        if self.kernel_type == AutotuneKernelType.DECODE_ATTENTION and autotune_level == AutotuneLevel.FORCE_AUTOTUNE:
            autotune_level = AutotuneLevel.ADAPTIVE_AUTOTUNE
            if not getattr(self, "_decode_force_autotune_logged", False):
                logger.info(
                    f"Decode attention kernel {self.kernel_name}: FORCE_AUTOTUNE is treated as ADAPTIVE_AUTOTUNE "
                    "to avoid repeated tuning across layers and reduce startup time. Existing configs are reused. "
                    f"To retune, delete the cached config files in '{self.cache_dir}' before restarting "
                    "with LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 or 2."
                )
                self._decode_force_autotune_logged = True

        rank_id = 0 if not dist.is_initialized() else get_global_rank()
        world_size = 1 if not dist.is_initialized() else get_global_world_size()

        static_key = frozendict(self._static_key(*args, **kwargs))
        run_key = str(self._run_key(*args, **kwargs))

        # Lazy load the cached configs in lightllm/common/triton_utils/autotune_kernel_configs
        # 先尝试加载缓存；关闭已有配置预热时仍须正常读取配置，不能用开关短路缓存加载。
        if (self._try_load_cache(static_key) or Autotuner.is_autotune_warmup()) and self.warmup_all_exist_config:
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

        if Autotuner.is_kernel_autotune_warmup(self.kernel_type) and autotune_level in [
            AutotuneLevel.ADAPTIVE_AUTOTUNE,
            AutotuneLevel.FORCE_AUTOTUNE,
        ]:
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

    @property
    def cache_dir(self) -> str:
        if not hasattr(self, "_cache_dir"):
            device_name = get_current_device_name()
            if device_name is None:
                raise RuntimeError(
                    f"Autotuner for kernel {self.kernel_name} requires a visible CUDA/MUSA device "
                    f"to resolve its cache directory, but torch.cuda.is_available() is False."
                )
            self._cache_dir = os.path.join(
                Path(__file__).parent,
                "autotune_kernel_configs",
                get_triton_version(),
                device_name,
                self.kernel_name,
            )
            os.makedirs(self._cache_dir, exist_ok=True)
        return self._cache_dir

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

        # 仅为本次调优重建输入，构造开销不计入计时，最终执行仍使用调用方的原始输入。
        if self.rebuild_input_func is not None:
            args, kwargs = self.rebuild_input_func(*args, **kwargs)

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
            # 配置更新后，清除该 static_key 下缓存的旧匹配结果，避免继续使用旧配置，使新调优配置生效。
            # 新配置也可能改变其他 run_key 的最近邻选择，因此需要清除整个 static_key 的匹配缓存。
            self.fast_match_configs.pop(_static_key, None)

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
                # 可选参数也能参与 key；调用方省略时使用算子函数声明的默认值。
                parameter = inspect.signature(self.fn).parameters.get(name)
                if parameter is None or parameter.default is inspect.Parameter.empty:
                    raise KeyError(f"Missing argument '{name}' required by key function")
                values.append(parameter.default)
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
