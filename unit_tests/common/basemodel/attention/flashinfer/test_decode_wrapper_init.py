import sys
import weakref
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lightllm.common.basemodel.attention.flashinfer import fp, fp8, mla
from lightllm.common.basemodel.attention.flashinfer.utils import should_init_decode_wrapper
from lightllm.common.basemodel.infer_struct import InferStateInfo


def test_decode_wrapper_rejects_conflicting_graph_flags():
    infer_state = InferStateInfo()
    infer_state.is_cuda_graph_capture = True
    infer_state.is_cuda_graph_replay = True

    with pytest.raises(ValueError, match="capture=True, replay=True"):
        should_init_decode_wrapper(infer_state)


@pytest.fixture(
    params=[fp.FlashInferDecodeAttState, fp8.Fp8FlashInferDecodeAttState, mla.MlaFlashInferDecodeAttState],
    ids=["fp", "fp8", "mla"],
)
def decode_init(request, monkeypatch):
    state_cls = request.param
    module = mla if state_cls is mla.MlaFlashInferDecodeAttState else fp
    page_size = 1 if state_cls is fp8.Fp8FlashInferDecodeAttState else 4
    max_page_num = 64 // page_size
    wrapper = SimpleNamespace(plan=Mock())
    create_wrapper = Mock(return_value=wrapper)
    flashinfer = ModuleType("flashinfer")
    flashinfer.decode = SimpleNamespace(BatchDecodeWithPagedKVCacheWrapper=create_wrapper)
    flashinfer.mla = SimpleNamespace(BatchMLAPagedAttentionWrapper=create_wrapper)
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)

    def cpu_arange(*args, **kwargs):
        kwargs["device"] = "cpu"
        return torch.arange(*args, **kwargs)

    monkeypatch.setattr(module, "torch", SimpleNamespace(int32=torch.int32, arange=cpu_arange, empty=torch.empty))

    def cumsum_params(q_lengths, kv_lengths):
        return tuple(
            torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0, dtype=torch.int32)))
            for lengths in (q_lengths, kv_lengths)
        )

    monkeypatch.setattr(module, "gen_cumsum_pad0_tensor", cumsum_params)

    def repack(**kwargs):
        kwargs["out_page_indices"].fill_(7)

    repack_indices = Mock(side_effect=repack)
    monkeypatch.setattr(module, "repack_kv_index", repack_indices)
    can_run = Mock(return_value=True)
    backend = SimpleNamespace(
        model=SimpleNamespace(
            graph=SimpleNamespace(can_run=can_run), graph_max_batch_size=4, graph_max_len_in_batch=64
        ),
        infer_page_size=page_size,
        max_page_num=max_page_num,
        kv_indices_buffer=[torch.empty(4 * max_page_num, dtype=torch.int32)],
        tp_q_head_num=4,
        tp_kv_head_num=2,
        head_dim=8,
        kv_lora_rank=8,
        qk_rope_head_dim=2,
        softmax_scale=0.25,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float8_e4m3fn if page_size == 1 else torch.bfloat16,
        workspace_buffer_key="decode_init_test",
        workspace_buffer_size=1,
        get_gpu_workspace_buffer=Mock(return_value=torch.empty(1, dtype=torch.uint8)),
    )
    return SimpleNamespace(
        module=module,
        state=state_cls(backend=backend),
        backend=backend,
        wrapper=wrapper,
        create_wrapper=create_wrapper,
        repack_indices=repack_indices,
        can_run=can_run,
    )


@pytest.mark.parametrize("mode", ["eager", "capture", "replay"])
def test_decode_init_uses_explicit_graph_mode_for_wrapper_creation(decode_init, mode):
    infer_state = InferStateInfo()
    infer_state.input_ids = torch.tensor([1, 2], dtype=torch.int64)
    infer_state.batch_size = 2
    infer_state.b_seq_len = torch.tensor([5, 2], dtype=torch.int32)
    infer_state.max_kv_seq_len = 5
    infer_state.b_req_idx = torch.tensor([2, 3], dtype=torch.int32)
    infer_state.microbatch_index = 0
    infer_state.req_manager = SimpleNamespace(req_to_token_indexs=torch.zeros((4, 64), dtype=torch.int32))
    if mode == "capture":
        infer_state.is_cuda_graph_capture = True
    elif mode == "replay":
        infer_state.is_cuda_graph_replay = True

    decode_init.state._infer_state_weak_ref = weakref.ref(infer_state)
    decode_init.state.init_state()

    decode_init.can_run.assert_not_called()
    expected_starts = [0, 5, 7] if decode_init.backend.infer_page_size == 1 else [0, 2, 3]
    torch.testing.assert_close(decode_init.state.kv_starts, torch.tensor(expected_starts, dtype=torch.int32))
    assert decode_init.state.kv_indices.data_ptr() == decode_init.backend.kv_indices_buffer[0].data_ptr()
    assert torch.all(decode_init.state.kv_indices == 7)
    decode_init.repack_indices.assert_called_once()
    repack_args = decode_init.repack_indices.call_args.kwargs
    assert repack_args["b_req_idx"] is infer_state.b_req_idx
    assert repack_args["b_token_len"] is infer_state.b_seq_len
    if decode_init.module is fp:
        expected_last_page_lengths = [1, 1] if decode_init.backend.infer_page_size == 1 else [1, 2]
        torch.testing.assert_close(
            decode_init.state.kv_last_page_len_buffer, torch.tensor(expected_last_page_lengths, dtype=torch.int32)
        )
    else:
        torch.testing.assert_close(decode_init.state.q_indptr, torch.tensor([0, 1, 2], dtype=torch.int32))

    if mode == "replay":
        assert decode_init.state.decode_wrapper is None
        decode_init.create_wrapper.assert_not_called()
        decode_init.backend.get_gpu_workspace_buffer.assert_not_called()
    else:
        assert decode_init.state.decode_wrapper is decode_init.wrapper
        decode_init.create_wrapper.assert_called_once()
        decode_init.backend.get_gpu_workspace_buffer.assert_called_once()
        decode_init.wrapper.plan.assert_called_once()
