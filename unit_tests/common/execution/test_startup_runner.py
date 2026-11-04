from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention_state.allocators import full_att_kv_allocator as allocator_module
from lightllm.common.attention_state.allocators import FullAttKVAllocator
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.core.runner.state import BatchStateFactory
from lightllm.core.runner.warmup import WarmupCleanupError, WarmupRunner
from lightllm.common.req_manager.base import ReqManager, _ReqLinkedList
from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType


def make_resources(monkeypatch, *, device="cpu", page_size=4):
    monkeypatch.setattr(allocator_module, "SharedInt", lambda _: SimpleNamespace(set_value=lambda value: None))
    monkeypatch.setattr(allocator_module, "get_current_rank_in_node", lambda: 0)
    allocator = FullAttKVAllocator(64, page_size=page_size, pin_memory=False)
    requests = ReqManager.__new__(ReqManager)
    requests.req_list = _ReqLinkedList(2)
    requests.max_request_num = requests.HOLD_REQUEST_ID = 2
    requests.page_size = page_size
    requests.req_to_token_indexs = torch.full((3, 32), -1, dtype=torch.int32, device=device)
    requests.bind_hold_page(allocator.hold_token_indices)
    requests.free_all = allocator.free_all = lambda: pytest.fail("reset of shared allocation domain")
    return requests, allocator


def make_runner(requests, allocator, forward, *, runtime=None, overlap=None):
    return WarmupRunner(
        requests,
        allocator,
        forward,
        lambda tokens: {"mtp_draft_input_hiddens": torch.ones(tokens, 4)},
        19,
        overlap_prefill=overlap,
        overlap_decode=overlap,
        runtime=runtime or SimpleNamespace(synchronize=lambda: None, barrier=lambda: None, empty_cache=lambda: None),
    )


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_state_factory_binds_resources_and_preserves_lane_and_model_fields(monkeypatch, prefill, legacy):
    requests, allocator = make_resources(monkeypatch)
    runner = make_runner(requests, allocator, None)
    batch = runner.prefill_input(3, 0) if prefill else runner.decode_input(3, 16)
    if not prefill:
        batch.b_position_delta.copy_(torch.tensor([1, 2, 3]))
        batch.b_shared_seq_len.copy_(torch.tensor([4, 5, 6]))
        batch.b_shared_radix_node_id.copy_(torch.tensor([7, 8, 9]))
    indices = torch.tensor([10, 11, 12])
    groups = (object(), object())
    execution = None if legacy else object()
    created = []

    def create_legacy(infer_state):
        assert infer_state.mem_index is indices
        assert infer_state.req_manager is requests
        created.append(infer_state)
        return SimpleNamespace(owner=infer_state)

    field = "prefill_att_state" if prefill else "decode_att_state"
    factories = tuple((field + suffix, create_legacy) for suffix in ("", "1")) if legacy else ()
    factory = BatchStateFactory(
        InferStateInfo,
        SimpleNamespace(new_instance=object),
        requests,
        allocator,
        "draft1",
        True,
        True,
        groups.__getitem__,
        execution_runner=execution,
        prefill_states=factories if prefill else (),
        decode_states=factories if not prefill else (),
        select_indices=lambda value: indices,
    )
    state = factory.create(batch, 1)
    assert state.input_ids is batch.input_ids and state.b_req_idx is batch.b_req_idx
    assert state.mtp_draft_input_hiddens is batch.mtp_draft_input_hiddens
    assert state.multimodal_params is batch.multimodal_params
    assert state.att_state_role == "draft1" and state.is_draft_model and state.return_all_prompt_logics
    assert state.execution_runner is execution and state.att_state_manager is allocator
    assert state.dist_group is groups[1] and state.microbatch_index == 1
    assert state.mem_index is indices
    assert len(created) == (2 if legacy else 0)
    if prefill:
        assert state.b_ready_cache_len is batch.b_ready_cache_len
    else:
        for name in ("b_position_delta", "b_shared_seq_len", "b_shared_radix_node_id"):
            assert getattr(state, name) is getattr(batch, name)
    other = factory.create(batch, 0)
    assert other.hidden_collector is not state.hidden_collector
    assert other.dist_group is groups[0]


@pytest.mark.parametrize("failure", [None, "forward", "allocation", "synchronize"])
def test_owned_warmup_releases_only_completed_resources(monkeypatch, failure):
    requests, allocator = make_resources(monkeypatch)
    live = requests.alloc()
    live_indices = allocator.alloc(4).clone()
    requests.req_to_token_indexs[live, :4] = live_indices
    live_mapping = requests.req_to_token_indexs[live].clone()
    submitted = []

    def synchronize():
        assert requests.req_list.marks[live] == 1
        if failure == "synchronize":
            raise RuntimeError("device failure")

    runner = make_runner(
        requests,
        allocator,
        None,
        runtime=SimpleNamespace(synchronize=synchronize, barrier=lambda: None, empty_cache=lambda: None),
    )
    if failure == "allocation":
        monkeypatch.setattr(allocator, "alloc", lambda _: (_ for _ in ()).throw(RuntimeError("allocation failure")))

    def run():
        with runner.allocated_prefill(5, random=True) as batch:
            assert batch.input_ids.min() >= 0 and batch.input_ids.max() < 19
            assert batch.b_req_idx.item() != live
            assert allocator.can_use_mem_size == 52  # 4 live + 8 page-aligned warmup tokens.
            submitted.append(batch)
            if failure == "forward":
                raise RuntimeError("forward failure")

    if failure:
        expected = WarmupCleanupError if failure == "synchronize" else RuntimeError
        with pytest.raises(expected):
            run()
    else:
        run()
    torch.testing.assert_close(requests.req_to_token_indexs[live], live_mapping)
    if failure == "synchronize":
        assert requests.req_list.marks == [1, 1]
        assert allocator.can_use_mem_size == 52
    else:
        assert requests.req_list.marks == [1, 0]
        assert allocator.can_use_mem_size == 60


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("prefill", [False, True])
def test_graph_warmup_uses_only_hold_rows(monkeypatch, overlap, prefill):
    requests, allocator = make_resources(monkeypatch)
    requests.alloc()
    allocator.alloc(4)
    batches = []

    def forward(*inputs):
        batches.append(inputs)
        for batch in inputs:
            assert batch.is_prefill is prefill
            assert (batch.b_req_idx == requests.HOLD_REQUEST_ID).all()
            assert batch.mtp_draft_input_hiddens.shape == (batch.input_ids.numel(), 4)
            if not prefill:
                assert (batch.b_seq_len == 2).all() and batch.max_kv_seq_len == 16
                assert (batch.b_position_delta == 0).all()

    runner = make_runner(requests, allocator, forward, overlap=forward)
    graph = SimpleNamespace(
        cuda_graph_batch_sizes=[3, 6],
        graph_max_len_in_batch=16,
        max_batch_size=6,
        graph_handle_token_nums=[4, 8],
        max_handle_token_num=8,
    )
    if prefill:
        runner.capture_prefill(graph, overlap=overlap)
    else:
        runner.capture_decode(graph, overlap=overlap)
    assert [len(inputs) for inputs in batches] == [2 if overlap else 1] * 2
    assert [inputs[0].input_ids.numel() for inputs in batches] == ([8, 4] if prefill else [6, 3])
    assert requests.req_list.marks == [1, 0]
    assert allocator.can_use_mem_size == 60


def test_autotune_restores_model_scope_and_phase_after_failure(monkeypatch):
    requests, allocator = make_resources(monkeypatch)
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.layers_num = 7
    model.autotune_layers = lambda: 2
    attempts = []

    def forward(batch):
        assert model.layers_num == 2
        assert Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.GENERAL)
        attempts.append(batch.input_ids.numel())
        raise RuntimeError("recoverable warmup failure")

    runner = make_runner(requests, allocator, forward)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        runner.autotune(4, model._autotune_recipe)
        assert Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.DECODE_ATTENTION)
    assert model.layers_num == 7 and attempts == [4, 1]
    assert requests.req_list.is_all_free() and allocator.can_use_mem_size == 64


def test_autotune_stops_after_unknown_completion(monkeypatch):
    requests, allocator = make_resources(monkeypatch)
    attempts = []

    @contextmanager
    def scope():
        yield

    def synchronize():
        raise RuntimeError("device failure")

    runner = make_runner(
        requests,
        allocator,
        lambda batch: attempts.append(batch.input_ids.numel()),
        runtime=SimpleNamespace(synchronize=synchronize, barrier=lambda: None, empty_cache=lambda: None),
    )
    with pytest.raises(WarmupCleanupError):
        runner.autotune(4, scope)
    assert attempts == [4]
    assert not Autotuner.is_autotune_warmup()
    assert allocator.can_use_mem_size == 60 and requests.req_list.marks == [1, 0]
