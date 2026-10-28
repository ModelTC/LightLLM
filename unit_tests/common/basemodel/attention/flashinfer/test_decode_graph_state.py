import sys
import weakref
from types import ModuleType, SimpleNamespace

import torch

from lightllm.common.basemodel.attention.flashinfer.fp import FlashInferDecodeAttState
from lightllm.common.basemodel.attention.flashinfer import mla
from lightllm.common.basemodel.infer_struct import InferStateInfo


def test_fp_graph_copy_refreshes_plan_with_runtime_length_and_preserves_buffers(monkeypatch):
    plan_calls = []
    flashinfer = ModuleType("flashinfer")
    decode = ModuleType("flashinfer.decode")
    decode.fast_decode_plan = lambda wrapper, **kwargs: plan_calls.append((wrapper, kwargs))
    flashinfer.decode = decode
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.decode", decode)
    backend = SimpleNamespace(
        infer_page_size=16,
        tp_q_head_num=4,
        tp_kv_head_num=2,
        head_dim=8,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    captured_owner = InferStateInfo()
    captured_owner.batch_size = 2
    runtime_owner = InferStateInfo()
    runtime_owner.max_kv_seq_len = 17
    captured = FlashInferDecodeAttState(
        backend=backend,
        _infer_state_weak_ref=weakref.ref(captured_owner),
        kv_starts=torch.tensor([0, 4, 8], dtype=torch.int32),
        kv_indices=torch.arange(8, dtype=torch.int32),
        kv_last_page_len_buffer=torch.tensor([16, 16], dtype=torch.int32),
        decode_wrapper=object(),
    )
    runtime = FlashInferDecodeAttState(
        backend=backend,
        _infer_state_weak_ref=weakref.ref(runtime_owner),
        kv_starts=torch.tensor([0, 2, 3], dtype=torch.int32),
        kv_indices=torch.arange(7, -1, -1, dtype=torch.int32),
        kv_last_page_len_buffer=torch.tensor([1, 5], dtype=torch.int32),
    )
    buffer_names = ("kv_starts", "kv_indices", "kv_last_page_len_buffer")
    captured_ptrs = {name: getattr(captured, name).data_ptr() for name in buffer_names}

    captured.copy_for_decode_cuda_graph(runtime)

    assert len(plan_calls) == 1
    wrapper, plan = plan_calls[0]
    assert wrapper is captured.decode_wrapper
    assert plan["indptr"] is captured.kv_starts
    assert plan["indices"] is captured.kv_indices
    assert plan["last_page_len"] is captured.kv_last_page_len_buffer
    torch.testing.assert_close(plan["global_override_indptr_cpu"], torch.tensor([0, 2, 4], dtype=torch.int32))
    for name in buffer_names:
        assert getattr(captured, name).data_ptr() == captured_ptrs[name]
        torch.testing.assert_close(getattr(captured, name), getattr(runtime, name))


def test_mla_graph_copy_refreshes_plan_with_runtime_lengths_and_preserves_buffers(monkeypatch):
    plan_calls = []
    monkeypatch.setattr(mla, "fill_mla_decode_plan_for_cuda_graph", lambda **kwargs: plan_calls.append(kwargs))
    backend = SimpleNamespace(tp_q_head_num=4)
    captured_owner = InferStateInfo()
    captured_owner.batch_size = 2
    runtime_owner = InferStateInfo()
    runtime_owner.b_seq_len = torch.tensor([17, 5], dtype=torch.int32)
    runtime_owner.max_kv_seq_len = 17
    captured = mla.MlaFlashInferDecodeAttState(
        backend=backend,
        _infer_state_weak_ref=weakref.ref(captured_owner),
        kv_starts=torch.tensor([0, 4, 8], dtype=torch.int32),
        kv_indices=torch.arange(8, dtype=torch.int32),
        decode_wrapper=object(),
    )
    runtime = mla.MlaFlashInferDecodeAttState(
        backend=backend,
        _infer_state_weak_ref=weakref.ref(runtime_owner),
        kv_starts=torch.tensor([0, 2, 3], dtype=torch.int32),
        kv_indices=torch.arange(7, -1, -1, dtype=torch.int32),
    )
    buffer_names = ("kv_starts", "kv_indices")
    captured_ptrs = {name: getattr(captured, name).data_ptr() for name in buffer_names}

    captured.copy_for_decode_cuda_graph(runtime)

    assert len(plan_calls) == 1
    plan = plan_calls[0]
    assert plan["decode_wrapper"] is captured.decode_wrapper
    assert plan["kv_page_indptr"] is captured.kv_starts
    assert plan["batch_size"] == 2
    assert plan["max_kv_len"] == 17
    torch.testing.assert_close(plan["kv_lens"], torch.tensor([17, 5], dtype=torch.int32))
    for name in buffer_names:
        assert getattr(captured, name).data_ptr() == captured_ptrs[name]
        torch.testing.assert_close(getattr(captured, name), getattr(runtime, name))
