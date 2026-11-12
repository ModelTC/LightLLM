import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lightllm.common.attention.backends.dense.flashinfer import backend as fp
from lightllm.common.attention.backends.dense.flashinfer.fp8 import Fp8FlashInferAttBackend
from lightllm.common.attention.backends.mla.flashinfer import backend as mla
from unit_tests.common.attention.helpers import make_backend, make_batch, make_context


@pytest.fixture(params=[fp.FlashInferAttBackend, Fp8FlashInferAttBackend, mla.MlaFlashInferAttBackend])
def decode_case(request, monkeypatch):
    cls = request.param
    module = mla if cls is mla.MlaFlashInferAttBackend else fp
    wrapper = SimpleNamespace(plan=Mock())
    create = Mock(return_value=wrapper)
    flashinfer = ModuleType("flashinfer")
    decode = ModuleType("flashinfer.decode")
    decode.BatchDecodeWithPagedKVCacheWrapper = create
    decode.fast_decode_plan = Mock()
    flashinfer.decode = decode
    flashinfer.mla = SimpleNamespace(BatchMLAPagedAttentionWrapper=create)
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.decode", decode)

    def cumsum(q, kv):
        return tuple(torch.cat((torch.zeros(1, dtype=torch.int32), v.cumsum(0).int())) for v in (q, kv))

    monkeypatch.setattr(module, "gen_cumsum_pad0_tensor", cumsum)
    repack = Mock(side_effect=lambda **kwargs: kwargs["out_page_indices"].fill_(int(kwargs["b_token_len"][0])))
    monkeypatch.setattr(module, "repack_kv_index", repack)
    mla_refresh = Mock()
    monkeypatch.setattr(mla, "fill_mla_decode_plan_for_cuda_graph", mla_refresh)
    page_size = 1 if cls is Fp8FlashInferAttBackend else 4
    backend = make_backend(
        cls,
        infer_page_size=page_size,
        max_page_num=64 // page_size,
        tp_q_head_num=4,
        tp_kv_head_num=2,
        head_dim=8,
        kv_lora_rank=8,
        qk_rope_head_dim=2,
        softmax_scale=0.25,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        workspace_buffer_key="test",
        workspace_buffer_size=8,
    )
    return SimpleNamespace(
        backend=backend,
        create=create,
        wrapper=wrapper,
        repack=repack,
        refresh=mla_refresh if module is mla else decode.fast_decode_plan,
        mla=module is mla,
    )


@pytest.mark.parametrize("capture", [False, True])
def test_decode_context_creates_wrapper_once_with_explicit_graph_mode(decode_case, capture):
    case = decode_case
    batch = make_batch()
    context = make_context(case.backend, batch, capture=capture, max_length=64, prepare=True)
    assert context.decode_wrapper is case.wrapper
    case.create.assert_called_once()
    assert case.create.call_args.kwargs["use_cuda_graph"] is capture
    assert context.kv_starts.tolist() == ([0, 5, 7] if case.backend.infer_page_size == 1 else [0, 2, 3])
    case.wrapper.plan.assert_called_once()
    case.repack.assert_called_once()


def test_graph_replay_refreshes_existing_wrapper_with_current_lengths(decode_case):
    case = decode_case
    batch = make_batch()
    context = make_context(case.backend, batch, capture=True, max_length=64, prepare=True)
    pointers = (context.kv_starts.data_ptr(), context.kv_indices.data_ptr(), context.batch.b_seq_len.data_ptr())
    for lengths in ((17, 5), (1, 1), (60, 3)):
        runtime = make_batch(lengths, req_to_token_indexs=batch.req_to_token_indexs)
        case.backend.prepare(context, runtime)
        assert pointers == (
            context.kv_starts.data_ptr(),
            context.kv_indices.data_ptr(),
            context.batch.b_seq_len.data_ptr(),
        )
        assert context.batch.b_seq_len.tolist() == list(lengths)
        assert context.kv_indices[0] == lengths[0]
        args = case.refresh.call_args.kwargs
        if case.mla:
            assert args["max_kv_len"] == max(lengths)
            assert args["decode_wrapper"] is case.wrapper
        else:
            pages = (max(lengths) + case.backend.infer_page_size - 1) // case.backend.infer_page_size
            assert args["global_override_indptr_cpu"].tolist() == [0, pages, 2 * pages]
    case.create.assert_called_once()
    assert case.refresh.call_count == 3
