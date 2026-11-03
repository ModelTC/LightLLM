import gc
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch

from lightllm.common.attention_state.memory_plan import merge_memory_demands
from lightllm.common.device_runtime import get_device_runtime
from lightllm.utils.log_utils import init_logger
from lightllm.utils.torch_memory_saver_utils import MemoryTag

logger = init_logger(__name__)


@dataclass(frozen=True)
class MemoryObservation:
    stage: str
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    device_used_bytes: int | None


class StartupMemoryProfile:
    def __init__(self, runtime=None):
        self.runtime = runtime or get_device_runtime("cuda")
        self.runtime.synchronize()
        initial = self.runtime.memory_stats()
        self.initial_allocated = initial["allocated_bytes"]
        self.initial_reserved = initial["reserved_bytes"]
        self.initial_device_used = initial["device_used_bytes"]
        self.observations = []

    @contextmanager
    def stage(self, name):
        self.runtime.reset_memory_peak()
        try:
            yield
        finally:
            self.runtime.synchronize()
            self.observations.append(MemoryObservation(name, **self.runtime.memory_stats()))

    @property
    def peak_bytes(self):
        changes = [0]
        for item in self.observations:
            for current, initial in (
                (item.allocated_bytes, self.initial_allocated),
                (item.reserved_bytes, self.initial_reserved),
                (item.peak_allocated_bytes, self.initial_allocated),
                (item.peak_reserved_bytes, self.initial_reserved),
                (item.device_used_bytes, self.initial_device_used),
            ):
                if current is not None and initial is not None:
                    changes.append(current - initial)
        return max(changes)

    def explain(self):
        return dict(
            peak_bytes=self.peak_bytes,
            peak_tracking=self.runtime.tracks_memory_peaks,
            stages=[asdict(item) for item in self.observations],
        )


def _distributed():
    return torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1


def _release_unpublished_runtime(models, manager):
    models[0].runtime.synchronize()
    for model in models:
        model.release_attention_runtime()
    manager.release_unpublished_storage()
    models[0].req_manager.free_all()
    gc.collect()
    models[0].runtime.empty_cache()


def initialize_attention_runtime(models):

    main = models[0]
    runtime = main.runtime
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
                device=runtime.device,
                runtime_demands=demands,
                runtime_peak_bytes=runtime_reserve,
            )
        profile = StartupMemoryProfile(runtime)
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
        planned_workspace = sum(d.fixed_bytes for d in demands if d.domain == runtime.device.type)
        observed_runtime = max(0, profile.peak_bytes - planned_workspace)
        reserved = next((d.fixed_bytes for d in plan.demands if d.name == "runtime_headroom"), 0)
        domain = runtime.device.type
        measured_plan = plan.with_runtime_peak(observed_runtime, domain)
        shortage = max(0, measured_plan.usage(domain) - dict(plan.limits)[domain])
        if failure is not None:
            storage_bytes = sum(
                d.at_capacity(plan.capacity)
                for d in plan.demands
                if d.bytes_per_unit and d.domain == runtime.device.type
            )
            shortage = max(shortage, min(256 * 1024 ** 2, max(storage_bytes // 10, 1)))
        if _distributed():
            missing = torch.tensor(shortage, dtype=torch.int64, device=runtime.device)
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
