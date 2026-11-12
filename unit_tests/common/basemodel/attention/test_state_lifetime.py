import gc
import weakref
from dataclasses import replace

import pytest
import torch

from lightllm.common.attention.backends.base import BaseAttentionBackend
from unit_tests.common.attention.helpers import make_backend, make_batch, make_plan, model_stub


def test_plan_does_not_retain_runner_or_model():
    model = model_stub()
    backend = BaseAttentionBackend(model)
    assert not any(value is model for value in vars(backend).values())
    plan = make_plan(backend)
    gc.disable()
    try:
        plan_ref = weakref.ref(plan)
        batch_ref = weakref.ref(plan.batch)
        del plan
        assert plan_ref() is None
        assert batch_ref() is None
    finally:
        gc.enable()


def test_captured_plan_keeps_buffers_while_transient_batch_dies():
    backend = make_backend()
    plan = make_plan(backend, capture=True, max_length=64)
    pointer = plan.batch.b_seq_len.data_ptr()
    runtime = make_batch((17, 3), req_to_token_indexs=plan.batch.req_to_token_indexs)
    ref = weakref.ref(runtime)
    backend.prepare(plan, runtime)
    del runtime
    assert ref() is None
    assert plan.batch.b_seq_len.tolist() == [17, 3]
    assert plan.batch.b_seq_len.data_ptr() == pointer
    assert plan.current_max_kv_seq_len == 17
