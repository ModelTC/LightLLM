from __future__ import annotations

from unit_tests.common.att_resources.model_helpers import read_kv_storage

from unit_tests.common.att_resources.model_helpers import main_linear_state, prepare_model_states
from lightllm.common.att_resources.full_att import FullAttentionState
from unit_tests.common.att_resources.model_helpers import owned_states
import dataclasses

import pytest
import torch

from lightllm.common.req_manager import ReqManager
from lightllm.models.glm5_next.model import Glm5NextTpPartModel
from lightllm.models.glm5_next_mtp.model import Glm5NextMTPModel
from lightllm.models.qwen3next.model import Qwen3NextTpPartModel
from lightllm.models.qwen3_5_mtp.model import Qwen3_5MTPModel
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.envs_utils import (
    get_added_mtp_kv_layer_num,
    get_env_start_args,
    get_unique_server_name,
    set_env_start_args,
    set_unique_server_name,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def clear_mtp_capacity_cache():
    get_added_mtp_kv_layer_num.cache_clear()
    yield
    get_added_mtp_kv_layer_num.cache_clear()


@pytest.mark.parametrize("glm", [False, True])
@pytest.mark.parametrize("mtp", [0, 2])
@pytest.mark.parametrize("mode", ["normal", "prefill"])
def test_real_model_resource_hooks_and_shared_draft(monkeypatch: pytest.MonkeyPatch, glm: bool, mtp: int, mode: str):
    args = StartArgs(
        run_mode=mode,
        data_type="bfloat16",
        page_size=4,
        pd_kv_page_size=128,
        linear_att_hash_page_size=4,
        linear_att_page_block_num=2,
        linear_att_cache_size=2,
        cpu_cache_token_page_size=8,
        mtp_step=mtp,
        mtp_mode="eagle_with_att" if mtp else None,
    )
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_NODE", "0")
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_DP", "0")
    monkeypatch.setenv("LIGHTLLM_DP_WORLD_SIZE", "1")
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    monkeypatch.setattr("lightllm.common.req_manager.req_sampling_params.get_vocab_size", lambda _: 128)
    set_env_start_args(dataclasses.asdict(args))
    get_env_start_args.cache_clear()
    set_unique_server_name(args)
    get_unique_server_name.cache_clear()
    model_type = Glm5NextTpPartModel if glm else Qwen3NextTpPartModel
    model = model_type.__new__(model_type)
    model.args, model.data_type = args, torch.bfloat16
    model.tp_world_size_, model.tp_rank_ = 1, 0
    model.max_req_num, model.max_total_token_num = 3, None
    model.max_seq_length, model.batch_max_tokens = 32, 16
    model.mem_fraction, model.num_kv_heads = 0.9, 2
    model.config = {
        "num_attention_heads": 4,
        "hidden_size": 64,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "n_layer": 4,
        "num_hidden_layers": 4,
        "full_attention_interval": 2,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 3,
    }
    if glm:
        model.config.update(
            num_key_value_heads=1,
            kv_lora_rank=512,
            index_kpool=4,
            index_head_dim=128,
            layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
            linear_attn_config={"num_heads": 4, "head_dim": 8, "short_conv_kernel_size": 3, "kda_layers": [0, 1, 2]},
        )
    observed = []

    def check_runtime_before_profile(device: str | torch.device | int):
        resources = model.att_resource_manager
        states = resources.get_full_attention_states()
        assert main_linear_state(resources).conv_state.is_cuda
        assert main_linear_state(resources).big_pages_pool is None
        assert resources.full_att_allocator is None
        assert not hasattr(resources, "pd_page_bytes")
        assert not hasattr(resources, "pd_transfer_layouts")
        assert read_kv_storage(states[0]) is None
        observed.append(main_linear_state(resources).conv_state.data_ptr())
        reserve = 0
        if mode == "prefill":
            reserve = resources.get_pd_page_bytes() * args.pd_kv_page_num
        return resources.get_full_att_token_bytes() * (32 + args.page_size) + reserve, 0

    monkeypatch.setattr(torch.cuda, "mem_get_info", check_runtime_before_profile)
    model._init_req_manager()
    prepare_model_states(model)
    drafts = []
    if mtp:
        draft_type = Glm5NextMTPModel if glm else Qwen3_5MTPModel
        draft = draft_type.__new__(draft_type)
        draft.main_model, draft.mtp_previous_draft_models = model, []
        draft.args, draft.data_type = args, torch.bfloat16
        draft.tp_rank_, draft.tp_world_size_ = 0, 1
        draft.max_req_num, draft.max_total_token_num = model.max_req_num, None
        draft.config = dict(model.config, num_hidden_layers=1, n_layer=1, full_attention_interval=1)
        draft._init_req_manager()
        prepare_model_states(draft, logical_layer_ids=(4,))
        assert draft.att_resource_manager is model.att_resource_manager
        assert read_kv_storage(owned_states(draft.att_resource_manager.__dict__, FullAttentionState)[0]) is None
        drafts.append(draft)
    model.att_resource_manager.initialize_resources(
        args, token_capacity=model.max_total_token_num, mem_fraction=model.mem_fraction
    )
    model.req_manager.bind_hold_page(model.att_resource_manager.full_att_allocator.hold_token_indices)
    resources = model.att_resource_manager
    (full_state,) = resources.get_full_attention_states("main")
    assert type(model.req_manager) is ReqManager
    assert not hasattr(model.req_manager, "req_to_ssm_state")
    assert resources.full_att_allocator.size == full_state.capacity == 32
    assert not hasattr(model, "mem_manager")
    assert not hasattr(model, "full_att_state")
    assert full_state.config.logical_layer_groups == tuple(
        (layer,) for layer in model.linear_config.get_full_att_logical_layer_ids() if layer < model.config["num_hidden_layers"]
    )
    assert observed == [main_linear_state(resources).conv_state.data_ptr()]
    assert resources.bounded_att_big_page_allocator.size == 6
    assert resources.bounded_att_small_page_allocator.size == 2
    assert not hasattr(model, "mtp_state_cursors")
    assert (main_linear_state(resources).mtp_state_index is not None) == bool(mtp)
    if mode == "prefill":
        assert resources.get_pd_full_att_transfer_layouts()
        assert resources.get_pd_bounded_att_or_tail_transfer_layouts()
    if mtp:
        draft = drafts[0]
        assert draft.req_manager is model.req_manager
        assert draft.att_resource_manager is resources
        assert draft.att_resource_manager.full_att_allocator is resources.full_att_allocator
        assert not hasattr(draft, "full_att_state")
        (draft_state,) = resources.get_full_attention_states("draft0")
        assert draft_state is not full_state
        assert draft_state.capacity == full_state.capacity
        assert {layer for group in draft_state.config.logical_layer_groups for layer in group}.isdisjoint(
            layer for group in full_state.config.logical_layer_groups for layer in group
        )
        assert not hasattr(draft, "mem_manager")
        assert not hasattr(draft, "mtp_state_cursors")
        assert observed == [main_linear_state(resources).conv_state.data_ptr()]
