import gc
import weakref

from lightllm.common.attention.backends.base import BaseAttentionBackend
from unit_tests.common.attention.helpers import make_backend, make_batch, make_context, make_spec, model_stub


def test_context_does_not_retain_runner_or_model():
    model = model_stub()
    backend = BaseAttentionBackend(model, make_spec())
    assert not any(value is model for value in vars(backend).values())
    context = make_context(backend)
    gc.disable()
    try:
        batch_ref = weakref.ref(context.batch)
        del context
        assert batch_ref() is None
    finally:
        gc.enable()


def test_captured_context_keeps_buffers_while_transient_batch_dies():
    backend = make_backend()
    context = make_context(backend, capture=True, max_length=64)
    pointer = context.batch.b_seq_len.data_ptr()
    runtime = make_batch((17, 3), req_to_token_indexs=context.batch.req_to_token_indexs)
    ref = weakref.ref(runtime)
    backend.prepare(context, runtime)
    del runtime
    assert ref() is None
    assert context.batch.b_seq_len.tolist() == [17, 3]
    assert context.batch.b_seq_len.data_ptr() == pointer
    assert context.current_max_kv_seq_len == 17
