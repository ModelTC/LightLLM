from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention_state.manager import AttentionStateManager
from lightllm.common.attention_state.types import StateTransferLayout
from lightllm.models import get_model_class
from lightllm.models.draft_registry import get_draft_model_class
from lightllm.models.qwen4_exp.model import Qwen4ExpTpPartModel
from lightllm.models.qwen4_exp.mtp.model import Qwen4ExpMTPModel
from lightllm.models.qwen4_exp.attention_state import Qwen4ExpAttentionState
from lightllm.models.qwen4_exp.lexical_state import Qwen4ExpLexicalState, LexicalStateConfig
from lightllm.utils.envs_utils import get_env_start_args


@pytest.fixture
def lexical(monkeypatch):
    import lightllm.models.qwen4_exp.lexical_state as module

    def copy(*, copies):
        for source, destination in copies:
            destination.copy_(source)

    monkeypatch.setattr(module, "copy_bounded_att_tensors", copy)
    state = Qwen4ExpLexicalState(LexicalStateConfig((1, 5), 2, 6, 8, 63, torch.bfloat16), 4, 2, "cpu")
    state.state_name = "main_qwen4_lexical_state"
    state.initialize_page_pools(4, 2, pin_memory=False)
    return state


def test_lazy_registration_and_resource_factories(config):
    args = get_env_start_args()
    args.data_type, args.linear_att_ssm_data_type = "bf16", "float32"
    args.cpu_cache_token_page_size = 8
    args.pd_kv_page_size = 4
    for model_type in ("qwen4_exp", "qwen4_exp_text"):
        assert get_model_class({"model_type": model_type}) is Qwen4ExpTpPartModel
        assert get_draft_model_class({"model_type": model_type}, "eagle_with_att") is Qwen4ExpMTPModel
    manager = AttentionStateManager(request_capacity=4)
    for cls, role, layers in ((Qwen4ExpTpPartModel, "main", (0, 1, 2, 3)), (Qwen4ExpMTPModel, "draft0", (4,))):
        full = cls.create_full_att_states(config, args, layers)
        bounded = cls.create_bounded_att_states(config, args, layers, 4)
        for state in (*full, *bounded):
            manager.register_state(role, state)
    manager.finish_state_registration()
    state = manager.get_state("main", "", 3, Qwen4ExpAttentionState)
    assert isinstance(state, Qwen4ExpAttentionState)
    assert manager.get_state("draft0", "qwen4_gqa", 4) is not state
    assert len(manager.get_bounded_attention_states("main")) == 2
    assert not manager.get_bounded_attention_states("draft0")
    for state in manager.get_full_attention_states():
        regions = state.get_cpu_transfer_regions()
        assert len(regions) == 6
        assert all(left.byte_end <= right.byte_start for left, right in zip(regions, regions[1:]))
        assert {region.field for region in regions} == {
            "k",
            "v",
            "indexer.k",
            "indexer.v",
            "positions.k",
            "positions.v",
        }
        assert state.get_token_bytes() > 0


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_replicated_indexer_and_positions_layout(config, tp):
    args = get_env_start_args()
    args.data_type = "bf16"
    for rank in range(tp):
        (state,) = Qwen4ExpTpPartModel.create_full_att_states(config, args, (0, 1, 2, 3), rank, tp)
        regions = state._get_transfer_regions(8)
        for region in regions[2:]:
            assert region.write_enabled == (rank == 0)
            assert region.shape[2] == 1
        assert sum(region.byte_end - region.byte_start for region in regions) == regions[-1].byte_end


def test_state_storage_accounts_for_all_fields_and_restores_positions(config):
    args = get_env_start_args()
    args.data_type = "bf16"
    (state,) = Qwen4ExpTpPartModel.create_full_att_states(config, args, (0, 1, 2, 3))
    state.device = torch.device("cpu")
    state.allocate_storage(8)
    tensors = (
        state.k_storage,
        state.v_storage,
        state.indexer.k_storage,
        state.indexer.v_storage,
        state.positions.k_storage,
        state.positions.v_storage,
    )
    assert sum(t.numel() * t.element_size() for t in tensors) == (8 + state.page_size) * state.get_token_bytes()
    indices = torch.tensor([1, 4])
    cos = torch.arange(8).view(2, 4).bfloat16()
    state.store_positions(indices, cos, -cos)
    view = state.qsa_view(3)
    assert torch.equal(view.key_cos[indices], cos)
    assert torch.equal(view.key_sin[indices], -cos)
    assert view.raw_keys.data_ptr() == state.indexer.k_storage.data_ptr()


@pytest.mark.parametrize("page_kind", ["big", "small"])
def test_lexical_prefix_round_trip_is_independent_of_runtime(lexical, page_kind):
    lexical.ngram[:, 1].fill_(17)
    lexical.convolution[:, 1].fill_(3)
    lexical.save_bounded_att_runtime_to_pages([1], [128], page_kind, [0])
    lexical.reset_bounded_att_state([1])
    assert bool((lexical.ngram[:, 1] == 63).all())
    lexical.restore_bounded_att_runtime_from_pages([2], [128], page_kind, [0])
    assert bool((lexical.ngram[:, 2] == 17).all())
    assert bool((lexical.convolution[:, 2] == 3).all())
    lexical.reset_bounded_att_state([2])
    assert bool((lexical._pool(page_kind).ngram[0] == 17).all())


def layout_for(state):
    regions = state.get_cpu_transfer_regions()
    return StateTransferLayout(
        state.state_name,
        None,
        16,
        16 + regions[-1].byte_end,
        tuple(replace(region, byte_start=region.byte_start + 16, byte_end=region.byte_end + 16) for region in regions),
    )


def test_cpu_and_pd_lexical_transfers_preserve_unrelated_bytes(lexical):
    layout = layout_for(lexical)
    pages = torch.full((3, layout.byte_end + 16), 0x5A, dtype=torch.uint8)
    lexical.ngram[:, 1].fill_(7)
    lexical.convolution[:, 1].fill_(11)
    lexical.save_bounded_att_runtime_to_pages([1], [7], "big", [0])
    lexical.write_state_pages_to_cpu("big", [0, 0], pages, [1, 2], layout, [False, True])
    assert bool((pages[2] == 0x5A).all())
    assert bool((pages[:, :16] == 0x5A).all())
    assert bool((pages[:, layout.byte_end :] == 0x5A).all())
    lexical.read_state_pages_from_cpu(pages, [1], [2], layout)
    lexical.restore_bounded_att_runtime_from_pages([2], [7], "big", [2])
    assert torch.equal(lexical.ngram[:, 2], lexical.ngram[:, 1])
    assert torch.equal(lexical.convolution[:, 2], lexical.convolution[:, 1])
    lexical.write_to_pd_page(1, pages, 0, 7, layout)
    lexical.reset_bounded_att_state([2])
    lexical.read_from_pd_page(pages, 2, 0, 7, layout)
    assert torch.equal(lexical.ngram[:, 2], lexical.ngram[:, 1])
    assert torch.equal(lexical.convolution[:, 2], lexical.convolution[:, 1])


def test_dp_copy_includes_runtime_and_prefix(lexical):
    other = Qwen4ExpLexicalState(lexical.config, 4, 2, "cpu")
    other.initialize_page_pools(4, pin_memory=False)
    other.ngram[:, 0].fill_(29)
    other.convolution[:, 0].fill_(13)
    lexical.copy_runtime_state_from_dp([0], [2], other)
    assert bool((lexical.ngram[:, 2] == 29).all())
    other.save_bounded_att_runtime_to_pages([0], [8], "big", [1])
    lexical.copy_big_pages_from_dp([1], [3], other)
    lexical.restore_bounded_att_runtime_from_pages([1], [8], "big", [3])
    assert bool((lexical.convolution[:, 1] == 13).all())


def test_mtp_commit_maps_request_rows_and_keeps_hold(lexical):
    observed = []
    lexical.speculative_lexical_caches = {1: SimpleNamespace(commit=lambda *args: observed.append(args))}
    requests, accepted = torch.tensor([2, 0, 3]), torch.tensor([1, 3, 1])
    lexical.commit_bounded_att_state(requests, accepted)
    req, starts, lengths, hold = observed[0]
    assert req is requests and lengths is accepted
    assert starts.tolist() == [0, 1, 2] and hold == 3


def test_ipc_state_excludes_weights_and_speculative_workspaces(lexical):
    lexical.lexical_caches = {1: SimpleNamespace(layer_weight=object())}
    lexical.speculative_lexical_caches = {1: object()}
    exported = lexical.__getstate__()
    assert not exported["lexical_caches"] and not exported["speculative_lexical_caches"]
    assert exported["ngram"] is lexical.ngram
    assert exported["convolution"] is lexical.convolution
    assert lexical.lexical_caches and lexical.speculative_lexical_caches
