import gc
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch

from lightllm.common.attention_state.memory_plan import merge_memory_demands
from lightllm.utils.log_utils import init_logger
from lightllm.utils.torch_memory_saver_utils import MemoryTag

logger = init_logger(__name__)


@dataclass(frozen=True)
class MemoryObservation:
    stage: str
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    device_used_bytes: int


class StartupMemoryProfile:
    def __init__(self):
        torch.cuda.synchronize()
        self.initial_allocated = torch.cuda.memory_allocated()
        self.initial_reserved = torch.cuda.memory_reserved()
        free, total = torch.cuda.mem_get_info()
        self.initial_device_used = total - free
        self.observations = []

    @contextmanager
    def stage(self, name):
        torch.cuda.reset_peak_memory_stats()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            self.observations.append(
                MemoryObservation(
                    name,
                    torch.cuda.memory_allocated(),
                    torch.cuda.memory_reserved(),
                    torch.cuda.max_memory_allocated(),
                    torch.cuda.max_memory_reserved(),
                    total - free,
                )
            )

    @property
    def peak_bytes(self):

        return max(
            (
                value
                for item in self.observations
                for value in (
                    item.peak_allocated_bytes - self.initial_allocated,
                    item.peak_reserved_bytes - self.initial_reserved,
                    item.device_used_bytes - self.initial_device_used,
                )
            ),
            default=0,
        )

    def explain(self):
        return dict(peak_bytes=max(0, self.peak_bytes), stages=[asdict(item) for item in self.observations])


def _distributed():
    return torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1


def _release_unpublished_runtime(models, manager):
    torch.cuda.synchronize()
    for model in models:
        model.release_attention_runtime()
    manager.release_unpublished_storage()
    models[0].req_manager.free_all()
    gc.collect()
    torch.cuda.empty_cache()


def initialize_attention_runtime(models):

    main = models[0]
    manager = main.att_state_manager
    requested_capacity = main.max_total_token_num
    runtime_reserve = 0
    attempts = []
    for attempt in range(3):
        demands = merge_memory_demands(*(model.attention_memory_demands() for model in models))
        with main.torch_memory_saver.region(tag=MemoryTag.KV_CACHE):
            manager.initialize_storage(
                main.args,
                token_capacity=requested_capacity,
                mem_fraction=main.mem_fraction,
                runtime_demands=demands,
                runtime_peak_bytes=runtime_reserve,
            )
        profile = StartupMemoryProfile()
        failure = None
        try:
            for model in models:
                model.startup_memory_profile = profile
                with profile.stage(model.att_state_role + ":runtime"):
                    model.init_runtime()
        except torch.OutOfMemoryError as error:
            if _distributed():

                raise
            failure = str(error)
        finally:
            for model in models:
                model.startup_memory_profile = None

        plan = manager.memory_plan
        planned_workspace = sum(d.fixed_bytes for d in demands if d.domain == "cuda")
        observed_runtime = max(0, profile.peak_bytes - planned_workspace)
        reserved = next((d.fixed_bytes for d in plan.demands if d.name == "runtime_headroom"), 0)
        domain = "cuda" if "cuda" in dict(plan.limits) else "cpu"
        measured_plan = plan.with_runtime_peak(observed_runtime, domain)
        shortage = max(0, measured_plan.usage(domain) - dict(plan.limits)[domain])
        if failure is not None:
            storage_bytes = sum(
                d.at_capacity(plan.capacity) for d in plan.demands if d.bytes_per_unit and d.domain == "cuda"
            )
            shortage = max(shortage, min(256 * 1024 ** 2, max(storage_bytes // 10, 1)))
        if _distributed():
            missing = torch.tensor(shortage, dtype=torch.int64, device="cuda")
            torch.distributed.all_reduce(missing, op=torch.distributed.ReduceOp.MAX)
            shortage = missing.item()
        attempts.append(dict(capacity=plan.capacity, oom=failure, **profile.explain()))
        manager.startup_memory_attempts = tuple(attempts)
        logger.info("Attention startup memory: attempt=%d, %s", attempt + 1, attempts[-1])
        if not shortage:
            manager.memory_plan = measured_plan
            manager.storage_published = True
            return
        if requested_capacity is not None or attempt == 2:
            raise torch.OutOfMemoryError(
                f"attention startup exceeds its runtime reserve by {shortage} bytes at capacity {plan.capacity}; "
                "reduce max_total_token_num, batch_max_tokens or Graph limits"
            )
        runtime_reserve = max(observed_runtime, reserved + shortage)
        logger.info("Replanning unpublished attention storage with runtime reserve %d bytes", runtime_reserve)
        _release_unpublished_runtime(models, manager)
