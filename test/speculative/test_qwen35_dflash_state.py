from types import MethodType, SimpleNamespace

import pytest
import torch

from lightllm.common.speculative import BlockDraftLayout, SpeculativeConfig, get_block_draft_layout
from lightllm.models.qwen3_5_dflash.model import Qwen3_5DFlashModel
from lightllm.models.qwen3_dflash.layer_infer.transformer_layer_infer import (
    get_draft_layer_type,
)
from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextMemManager
from lightllm.server import api_start
from lightllm.utils import envs_utils, kv_cache_utils
from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl.prefill_impl import (
    PDChunkedPrefillForPrefillNode,
)
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.speculative.proposers.dflash import DFlashProposer


def test_block_layout_skips_leading_bonus_query_prediction():
    block_token_ids = torch.arange(32).reshape(2, 16)

    selected = DFlashProposer.select_draft_token_ids(
        block_token_ids=block_token_ids,
        draft_step=15,
        layout=BlockDraftLayout(query_block_size=16, proposal_output_start=1),
    )

    torch.testing.assert_close(selected, block_token_ids[:, 1:])


def test_block_layout_keeps_first_query_prediction():
    block_token_ids = torch.arange(32).reshape(2, 16)

    selected = DFlashProposer.select_draft_token_ids(
        block_token_ids=block_token_ids,
        draft_step=16,
        layout=BlockDraftLayout(query_block_size=16, proposal_output_start=0),
    )

    torch.testing.assert_close(selected, block_token_ids)


def test_block_draft_layout_defaults_to_zero_and_uses_architecture_override():
    draft_fields = {
        "block_size": 16,
        "target_layer_ids": [1, 10],
        "mask_token_id": 248077,
    }
    nested = {
        "architectures": ["Qwen3DFlashModel"],
        "dflash_config": draft_fields,
    }
    flat = {"architectures": ["Qwen3DFlashModel"], **draft_fields, "block_size": 7}
    qwen35 = {
        "architectures": ["Qwen3_5DFlashModel"],
        "dflash_config": draft_fields,
    }

    assert get_block_draft_layout(
        nested,
        mode="dflash",
    ) == BlockDraftLayout(query_block_size=16, proposal_output_start=0)
    assert get_block_draft_layout(
        flat,
        mode="dflash",
    ) == BlockDraftLayout(query_block_size=7, proposal_output_start=0)
    assert get_block_draft_layout(qwen35, mode="dflash") == BlockDraftLayout(
        query_block_size=16, proposal_output_start=1
    )


def test_qwen35_dflash_normalizes_block_size_to_fifteen_draft_tokens():
    backend = ModeBackend.__new__(ModeBackend)
    backend.args = SimpleNamespace(mtp_step=16)
    backend.mtp_step = 16
    backend.spec_config = SpeculativeConfig(mode="dflash", step=16)
    backend.logger = SimpleNamespace(warning=lambda *args: None)
    config = {
        "architectures": ["Qwen3_5DFlashModel"],
        "dflash_config": {
            "block_size": 16,
            "target_layer_ids": [1, 10],
            "mask_token_id": 248077,
        },
    }

    backend._normalize_block_mtp_step_from_config(config)

    assert backend.args.mtp_step == 15
    assert backend.mtp_step == 15
    assert backend.spec_config.step == 15
    assert backend.block_draft_layout == BlockDraftLayout(query_block_size=16, proposal_output_start=1)


def test_qwen35_dflash_respects_shorter_configured_draft_step():
    backend = ModeBackend.__new__(ModeBackend)
    backend.args = SimpleNamespace(mtp_step=9)
    backend.mtp_step = 9
    backend.spec_config = SpeculativeConfig(mode="dflash", step=9)
    backend.logger = SimpleNamespace(warning=lambda *args: None)
    config = {
        "architectures": ["Qwen3_5DFlashModel"],
        "dflash_config": {
            "block_size": 16,
            "target_layer_ids": [1, 10],
            "mask_token_id": 248077,
        },
    }

    backend._normalize_block_mtp_step_from_config(config)

    assert backend.args.mtp_step == 9
    assert backend.mtp_step == 9
    assert backend.spec_config.step == 9


def test_qwen35_dflash_normalizes_global_start_args_to_fifteen(monkeypatch):
    config = {
        "architectures": ["Qwen3_5DFlashModel"],
        "dflash_config": {
            "block_size": 16,
            "target_layer_ids": [1, 10],
            "mask_token_id": 248077,
        },
    }
    monkeypatch.setattr(
        api_start.PretrainedConfig,
        "get_config_dict",
        lambda *_args, **_kwargs: (config, {}),
    )
    args = SimpleNamespace(mtp_step=16, mtp_draft_model_dir=["unused"])

    spec_config = api_start.normalize_block_mtp_step_from_first_draft_config(
        args,
        SpeculativeConfig(mode="dflash", step=16),
    )

    assert args.mtp_step == 15
    assert spec_config.step == 15


def test_qwen35_dflash_global_normalization_respects_shorter_step(monkeypatch):
    config = {
        "architectures": ["Qwen3_5DFlashModel"],
        "dflash_config": {
            "block_size": 16,
            "target_layer_ids": [1, 10],
            "mask_token_id": 248077,
        },
    }
    monkeypatch.setattr(
        api_start.PretrainedConfig,
        "get_config_dict",
        lambda *_args, **_kwargs: (config, {}),
    )
    args = SimpleNamespace(mtp_step=9, mtp_draft_model_dir=["unused"])

    spec_config = api_start.normalize_block_mtp_step_from_first_draft_config(
        args,
        SpeculativeConfig(mode="dflash", step=9),
    )

    assert args.mtp_step == 9
    assert spec_config.step == 9


def test_qwen35_dflash_support_scope_preserves_existing_lightspec_modes(monkeypatch):
    backend = ModeBackend.__new__(ModeBackend)
    backend.is_linear_att_mixed_model = True
    backend.args = SimpleNamespace(mtp_draft_model_dir=["unused"])

    backend.spec_config = SpeculativeConfig(mode="qwen3next_eagle", step=2, dynamic_verify=True)
    backend._validate_linear_att_spec_support()

    backend.spec_config = SpeculativeConfig(mode="dflash", step=6)
    monkeypatch.setattr(
        "lightllm.server.router.model_infer.mode_backend.base_backend.PretrainedConfig.get_config_dict",
        lambda *_args, **_kwargs: ({"architectures": ["Qwen3DFlashModel"]}, {}),
    )
    with pytest.raises(AssertionError, match="Qwen3_5DFlashModel"):
        backend._validate_linear_att_spec_support()

    monkeypatch.setattr(
        "lightllm.server.router.model_infer.mode_backend.base_backend.PretrainedConfig.get_config_dict",
        lambda *_args, **_kwargs: ({"architectures": ["Qwen3_5DFlashModel"]}, {}),
    )
    backend.is_linear_att_mixed_model = False
    with pytest.raises(AssertionError, match="requires a Qwen3Next target"):
        backend._validate_linear_att_spec_support()

    backend.is_linear_att_mixed_model = True
    backend._validate_linear_att_spec_support()



def test_qwen35_dflash_restores_target_allocator_on_shared_req_manager():
    target_mem_manager = object()
    draft_mem_manager = object()
    req_manager = SimpleNamespace(mem_manager=draft_mem_manager)
    model = Qwen3_5DFlashModel.__new__(Qwen3_5DFlashModel)
    model.main_model = SimpleNamespace(req_manager=req_manager, mem_manager=target_mem_manager)
    model.req_manager = req_manager
    model.mem_manager = draft_mem_manager

    model._restore_main_mem_manager()

    assert model.req_manager.mem_manager is target_mem_manager
    assert model.mem_manager is draft_mem_manager


def test_qwen35_pd_move_page_supports_distinct_target_and_draft_kv_shapes():
    manager = Qwen3NextMemManager.__new__(Qwen3NextMemManager)
    manager.size = 16
    manager.dtype = torch.float32
    manager.layer_num = 2
    manager.head_dim = 4
    manager.linear_config = SimpleNamespace(full_att_all_num_kv_heads=3)
    manager._pd_dflash_draft_mem_manager = None
    manager._pd_dflash_global_kv_heads = None
    draft_manager = SimpleNamespace(
        size=16,
        dtype=torch.float32,
        layer_num=5,
        head_dim=2,
    )

    manager.register_dflash_draft_mem_manager(draft_manager, global_kv_heads=2)
    elements_per_token = max(2 * 2 * 3 * 4, 5 * 2 * 2 * 2)
    manager.kv_move_buffer = torch.empty((1, 7, 1, 1, elements_per_token))

    target_page = manager._get_pd_kv_page(0, "kv")
    draft_page = manager._get_pd_kv_page(0, "draft_kv")
    assert target_page.shape == (7, 2, 6, 4)
    assert draft_page.shape == (7, 5, 4, 2)
    assert target_page.is_contiguous()
    assert draft_page.is_contiguous()


def test_qwen35_pd_prefill_emits_target_and_draft_kv_tasks():
    emitted_page_kinds = []

    backend = PDChunkedPrefillForPrefillNode.__new__(PDChunkedPrefillForPrefillNode)
    backend.args = SimpleNamespace(pd_kv_page_size=4)
    backend.model = SimpleNamespace(mem_manager=SimpleNamespace(has_separate_dflash_draft_kv=True))
    backend.is_master_in_dp = False

    def fake_create_task(self, req_obj, kv_start_index, kv_end_index, page_kind="kv"):
        del self, req_obj, kv_start_index, kv_end_index
        emitted_page_kinds.append(page_kind)
        return SimpleNamespace(first_gen_token_id=None, first_gen_token_logprob=None)

    backend._create_pd_trans_task = MethodType(fake_create_task, backend)
    req = SimpleNamespace(
        cur_kv_len=4,
        pd_trans_kv_start_index=0,
        shm_req=SimpleNamespace(input_len=4),
    )

    backend._prefill_chuncked_handle_func(req, next_token_id=1, next_token_prob=0.0, output_len=0)

    assert emitted_page_kinds == ["kv", "draft_kv"]
    assert req.pd_trans_kv_start_index == 4


def test_dflash_layer_type_uses_draft_local_index_for_global_layer_numbers():
    config = {
        "_draft_layer_start": 64,
        "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
    }

    assert [get_draft_layer_type(i, config) for i in range(64, 70)] == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]



def test_qwen35_cpu_cache_appends_distinct_draft_kv_region(monkeypatch):
    page_num = 2
    page_size = 4
    target_bytes = 32
    manager = Qwen3NextMemManager.__new__(Qwen3NextMemManager)
    manager.size = 16
    manager.dtype = torch.float32
    manager.linear_config = SimpleNamespace(get_cpu_cache_big_page_bytes=lambda: target_bytes)
    manager._pd_dflash_draft_mem_manager = None
    manager._pd_dflash_global_kv_heads = None
    draft_manager = SimpleNamespace(
        size=16,
        dtype=torch.float32,
        layer_num=3,
        head_dim=2,
    )
    manager.register_dflash_draft_mem_manager(draft_manager, global_kv_heads=2)
    monkeypatch.setattr(
        "lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager.get_env_start_args",
        lambda: SimpleNamespace(cpu_cache_token_page_size=page_size),
    )

    draft_bytes = 3 * page_size * 4 * 2 * torch.float32.itemsize
    cpu_cache = torch.full((page_num, 1, 1, 1, target_bytes + draft_bytes), 0xA5, dtype=torch.uint8)
    draft_cache = manager.get_dflash_draft_cpu_cache(cpu_cache)

    assert draft_cache.shape == (page_num, 3, page_size, 4, 2)
    assert draft_cache.dtype == torch.float32
    draft_cache.zero_()
    assert torch.all(cpu_cache.reshape(page_num, -1)[:, :target_bytes] == 0xA5)
    assert torch.all(cpu_cache.reshape(page_num, -1)[:, target_bytes:] == 0)


def test_qwen35_dflash_does_not_reserve_duplicate_target_kv_layers(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        '{"architectures": ["Qwen3_5DFlashModel"], "n_layer": 5}',
        encoding="utf-8",
    )
    args = SimpleNamespace(
        mtp_mode="dflash",
        mtp_step=15,
        mtp_dynamic_verify=False,
        mtp_draft_model_dir=[str(tmp_path)],
    )
    monkeypatch.setattr(envs_utils, "get_env_start_args", lambda: args)
    envs_utils.get_added_mtp_kv_layer_num.cache_clear()

    assert envs_utils.get_added_mtp_kv_layer_num() == 0
    envs_utils.get_added_mtp_kv_layer_num.cache_clear()


def test_qwen35_dflash_cpu_cache_meta_includes_draft_kv(monkeypatch):
    args = SimpleNamespace(
        enable_cpu_cache=True,
        model_dir="unused-target",
        mtp_mode="dflash",
        mtp_step=15,
        mtp_dynamic_verify=False,
        cpu_cache_storage_size=1,
    )
    monkeypatch.setattr(kv_cache_utils, "get_env_start_args", lambda: args)
    monkeypatch.setattr(kv_cache_utils, "is_linear_att_mixed_model", lambda _path: True)
    monkeypatch.setattr(kv_cache_utils, "get_llm_data_type", lambda: torch.bfloat16)
    monkeypatch.setattr(
        kv_cache_utils.LinearAttCacheConfig,
        "load_from_args",
        lambda: SimpleNamespace(get_cpu_cache_big_page_bytes=lambda: 1024),
    )
    monkeypatch.setattr(kv_cache_utils, "_get_qwen35_dflash_cpu_cache_bytes", lambda _args: 384)
    kv_cache_utils.calcu_cpu_cache_meta.cache_clear()

    meta = kv_cache_utils.calcu_cpu_cache_meta()

    assert meta.data_type == torch.uint8
    assert meta.calcu_one_page_size() == 1024 + 384
    kv_cache_utils.calcu_cpu_cache_meta.cache_clear()


def test_qwen35_dflash_cpu_cache_draft_size_uses_checkpoint_shape(monkeypatch):
    from transformers.configuration_utils import PretrainedConfig

    monkeypatch.setattr(
        PretrainedConfig,
        "get_config_dict",
        lambda *_args, **_kwargs: ({"architectures": ["Qwen3_5DFlashModel"]}, {}),
    )
    monkeypatch.setattr(kv_cache_utils, "get_layer_num", lambda _path: 3)
    monkeypatch.setattr(kv_cache_utils, "get_num_key_value_heads", lambda _path: 2)
    monkeypatch.setattr(kv_cache_utils, "get_head_dim", lambda _path: 5)
    monkeypatch.setattr(kv_cache_utils, "get_llm_data_type", lambda: torch.bfloat16)
    args = SimpleNamespace(mtp_draft_model_dir=["unused-draft"], cpu_cache_token_page_size=4)

    draft_bytes = kv_cache_utils._get_qwen35_dflash_cpu_cache_bytes(args)

    assert draft_bytes == 4 * 3 * 2 * 2 * 5 * torch.bfloat16.itemsize
