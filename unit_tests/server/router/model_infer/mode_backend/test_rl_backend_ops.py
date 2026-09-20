from types import SimpleNamespace

import pytest

from lightllm.server.router.model_infer.mode_backend.rl_backend_ops import RlBackendOps
from lightllm.utils.torch_memory_saver_utils import MemoryTag


@pytest.mark.parametrize(
    ("tags", "should_init_hold_indexes"),
    [
        ([MemoryTag.WEIGHT], False),
        ([MemoryTag.KV_CACHE], True),
        (None, True),
    ],
)
def test_resume_memory_initializes_hold_indexes_only_for_kv_cache(tags, should_init_hold_indexes):
    events = []
    model = SimpleNamespace(
        torch_memory_saver=SimpleNamespace(resume=lambda tag: events.append(("resume", tag))),
        req_manager=SimpleNamespace(init_hold_request_indexs=lambda: events.append(("init_hold_indexes", None))),
    )
    backend = SimpleNamespace(model=model, logger=None)
    ops = RlBackendOps(backend)
    ops._clear_cuda_cache = lambda: None

    ops._resume_memory_tags(tags)

    init_events = [event for event in events if event[0] == "init_hold_indexes"]
    assert bool(init_events) is should_init_hold_indexes
    if should_init_hold_indexes:
        last_resume_index = max(index for index, event in enumerate(events) if event[0] == "resume")
        assert events.index(("init_hold_indexes", None)) > last_resume_index
