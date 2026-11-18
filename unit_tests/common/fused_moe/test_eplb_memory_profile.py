"""EPLB reserves post-profile buffers in the shared attention storage budget."""
import json
from types import SimpleNamespace

import pytest
import torch

from lightllm.common import eplb_utils
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state import manager as memory_module
from lightllm.common.basemodel import basemodel as base_module
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.expert_parallel_state import (
    EPLBState,
    ExpertParallelState,
)
from lightllm.models.deepseek2 import model as deepseek_module
from lightllm.models.deepseek_mtp.model import Deepseek3MTPModel
from lightllm.models.glm5_next_mtp.model import Glm5NextMTPModel
from lightllm.models.qwen3_moe.model import Qwen3MOEModel
from lightllm.models.qwen3_moe_mtp.model import Qwen3MOEMTPModel
from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
from lightllm.models.qwen3_5_mtp.model import Qwen3_5MTPModel
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils import device_utils


@pytest.fixture(autouse=True)
def supported_eplb_device(monkeypatch):
    monkeypatch.setattr(device_utils, "is_sm100_gpu", lambda: False)


@pytest.fixture
def profile_without_allocating(monkeypatch):
    budget = SimpleNamespace(free=40 * 1024**3, total=80 * 1024**3)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (budget.free, budget.total))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(memory_module, "get_current_device_id", lambda: 0)
    manager = AttentionStateManager(request_capacity=4)
    manager.finish_state_registration()
    manager.get_full_attention_states = lambda: [object()]
    manager.get_full_att_token_bytes = lambda: 1024
    manager.get_pd_operator = lambda: SimpleNamespace(get_page_bytes=lambda: 1024)
    allocated = {}
    manager.allocate_storage = lambda **kwargs: allocated.update(kwargs)
    return manager, allocated, budget


def make_expert(layer=0, redundant=0):
    rows = 64 + redundant
    expert = FusedMoeWeight.__new__(FusedMoeWeight)
    expert.enable_ep_moe = True
    expert.layer_num_ = layer
    expert.expert_parallel_state = ExpertParallelState(
        512,
        8,
        EPLBState(
            redundant,
            None,
            None,
            torch.empty((40, 512), dtype=torch.int64, device="meta"),
        ),
    )
    # Eight layers fit in an 80 GiB device; no actual large allocation is made.
    expert.w13 = SimpleNamespace(weight=torch.empty((rows, 6144, 4096), dtype=torch.bfloat16, device="meta"))
    expert.w2 = SimpleNamespace(weight=torch.empty((rows, 4096, 3072), dtype=torch.bfloat16, device="meta"))
    return expert


def make_model(model_class=Qwen3MOEModel, redundant=0):
    model = model_class.__new__(model_class)
    model.args = StartArgs(enable_ep_moe=True, enable_prefill_eplb=True, run_mode="prefill", mtp_mode=None)
    model.tp_world_size_ = 1  # DP8: the attention TP size is 1, while expert world size is 8.
    model.data_type = torch.bfloat16
    model.max_total_token_num = None
    model.mem_fraction = 0.98
    model.num_kv_heads = 8
    model.config = dict(
        hidden_size=4096,
        n_embed=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=8,
        n_layer=8,
        head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        full_attention_interval=4,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    model.trans_layers_weight = [SimpleNamespace(experts=make_expert(layer, redundant)) for layer in range(8)]
    return model


@pytest.mark.parametrize("redundant,expected", [(0, 36), (2, 60)])
@pytest.mark.parametrize("zero_point", [False, True])
def test_full_staging_includes_odd_rows_and_scalar_quantization_scales(redundant, expected, zero_point):
    expert = make_expert(redundant=redundant)
    expert.expert_parallel_state = ExpertParallelState(24, 8, expert.expert_parallel_state.eplb)
    rows = 3 + redundant
    expert.w13 = SimpleNamespace(
        weight=torch.empty((rows, 5), dtype=torch.uint8),
        weight_scale=torch.empty((rows, 1), dtype=torch.float32),
        weight_zero_point=torch.empty((rows, 1), dtype=torch.float32) if zero_point else None,
    )
    expert.w2 = SimpleNamespace(weight=torch.empty((rows, 3), dtype=torch.uint8))
    assert eplb_utils.get_eplb_staging_nbytes([expert] * 12) == expected + (4 * rows if zero_point else 0)


@pytest.mark.parametrize(
    "model_class,draft",
    [
        (base_module.TpPartBaseModel, False),
        (deepseek_module.Deepseek2TpPartModel, False),
        (Deepseek3MTPModel, True),
        (Qwen3MOEModel, False),
        (Qwen3MOEMTPModel, True),
        (Qwen3_5TpPartModel, False),
        (Qwen3_5MTPModel, True),
        (Glm5NextMTPModel, True),
    ],
)
def test_config_initialization_sets_model_role_before_weights(tmp_path, model_class, draft):
    config = dict(
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=16,
        num_hidden_layers=4,
        num_nextn_predict_layers=1,
        mhc=True,
        is_mtp_draft_model=not draft,
    )
    (tmp_path / "config.json").write_text(json.dumps(dict(config, text_config=config)))
    model = model_class.__new__(model_class)
    model.weight_dir_ = str(tmp_path)
    model.finetune_config = None
    model.tp_world_size_ = 1
    model.max_total_token_num = None
    model.main_model = SimpleNamespace(config=config)

    model._init_config()

    assert model.config["is_mtp_draft_model"] is draft
    assert config["is_mtp_draft_model"] is not draft
    assert config["mhc"] is True


@pytest.mark.parametrize("redundant", [0, 2])
@pytest.mark.parametrize("cell_bytes", [1024, 1280, 4096])
def test_shared_storage_reserves_eplb_before_kv_allocation(profile_without_allocating, redundant, cell_bytes):
    manager, allocated, budget = profile_without_allocating
    manager.get_full_att_token_bytes = lambda: cell_bytes
    model = make_model(redundant=redundant)
    reservations = eplb_utils.get_eplb_memory_reservations(model)
    original_fraction = model.mem_fraction
    model.args.page_size = 16
    model.args.pd_kv_page_num = 4

    manager.initialize_storage(
        model.args, token_capacity=None, mem_fraction=model.mem_fraction, device="cpu", memory_reservations=reservations
    )

    expected_budget = int(budget.free - budget.total * (1 - original_fraction)) - 4096 - sum(reservations.values())
    expected_tokens = (expected_budget // (cell_bytes * 16) - 1) * 16
    assert allocated["token_capacity"] == expected_tokens
    assert manager.memory_reservations == reservations
    assert model.mem_fraction == original_fraction
    assert expected_tokens * cell_bytes + cell_bytes * 16 + sum(reservations.values()) + 4096 <= budget.free


@pytest.mark.parametrize(
    "enabled,draft,run_mode", [(False, False, "prefill"), (True, True, "prefill"), (True, False, "decode")]
)
def test_disabled_eplb_and_draft_do_not_reserve_target_buffers(monkeypatch, enabled, draft, run_mode):
    model = make_model()
    model.args.enable_prefill_eplb = enabled
    model.args.run_mode = run_mode
    model.is_mtp_draft_model = draft
    monkeypatch.setattr(eplb_utils, "find_ep_moe_weights", lambda _: pytest.fail("budgeting must be skipped"))
    assert eplb_utils.get_eplb_memory_reservations(model) == {}


def test_weight_aliases_are_deduplicated_and_non_eplb_layers_are_excluded():
    model = make_model()
    first, last = model.trans_layers_weight[0].experts, model.trans_layers_weight[-1].experts
    ordinary = make_expert()
    ordinary.expert_parallel_state = ExpertParallelState(512, 8)
    model.trans_layers_weight = [
        SimpleNamespace(a=last, b=first, alias=first, ordinary=ordinary, dense=object()),
        SimpleNamespace(experts=last),
    ]
    weights = [first, last]
    assert eplb_utils.get_eplb_memory_reservations(model) == {
        "eplb_staging": eplb_utils.get_eplb_staging_nbytes(weights),
        "eplb_sampling": eplb_utils.get_eplb_sampling_peak_nbytes(weights),
    }


def test_explicit_shared_capacity_is_not_reduced_twice(profile_without_allocating):
    manager, allocated, _ = profile_without_allocating
    model = make_model()
    model.args.page_size = 16
    reservations = eplb_utils.get_eplb_memory_reservations(model)
    manager.initialize_storage(model.args, 1024, model.mem_fraction, device="cpu", memory_reservations=reservations)
    assert allocated["token_capacity"] == 1024


def test_shared_profile_uses_minimum_capacity_across_ranks(monkeypatch, profile_without_allocating):
    manager, allocated, _ = profile_without_allocating
    model = make_model()
    model.args.page_size = 16
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)
    original_tensor = torch.tensor
    monkeypatch.setattr(torch, "tensor", lambda data, *, dtype, device: original_tensor(data, dtype=dtype))

    def minimum_capacity(capacity, op):
        assert op == torch.distributed.ReduceOp.MIN
        assert capacity.item() > 1024
        capacity.fill_(1024)

    monkeypatch.setattr(torch.distributed, "all_reduce", minimum_capacity)
    manager.initialize_storage(
        model.args,
        None,
        model.mem_fraction,
        device="cpu",
        memory_reservations=eplb_utils.get_eplb_memory_reservations(model),
    )
    assert allocated["token_capacity"] == 1024


def test_insufficient_eplb_budget_is_rejected_before_allocation(profile_without_allocating):
    manager, allocated, budget = profile_without_allocating
    model = make_model()
    with pytest.raises(MemoryError, match="HOLD pages"):
        manager.initialize_storage(
            model.args, None, 1.0, device="cpu", memory_reservations={"eplb_staging": budget.free}
        )
    assert not allocated


def test_negative_reservation_is_rejected(profile_without_allocating):
    manager, allocated, _ = profile_without_allocating
    with pytest.raises(ValueError, match="non-negative"):
        manager.initialize_storage(StartArgs(), 1024, 0.9, device="cpu", memory_reservations={"eplb_staging": -1})
    assert not allocated
