"""EPLB reservations reach real KV profiling before any KV allocation."""
from types import SimpleNamespace

import pytest
import torch

from lightllm.common import eplb_utils
from lightllm.common.basemodel import basemodel as base_module
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.expert_parallel_state import (
    EPLBState,
    ExpertParallelState,
)
from lightllm.common.kv_cache_mem_manager import mem_manager as memory_module
from lightllm.common.kv_cache_mem_manager.deepseek2_mem_manager import Deepseek2MemoryManager
from lightllm.common.kv_cache_mem_manager.deepseek3_2mem_manager import Deepseek3_2MemoryManager
from lightllm.common.kv_cache_mem_manager.fp8_per_token_group_quant_deepseek3_2mem_manager import (
    FP8PerTokenGroupQuantDeepseek3_2MemoryManager,
)
from lightllm.common.kv_cache_mem_manager.fp8_static_per_head_quant_mem_manager import FP8StaticPerHeadQuantMemManager
from lightllm.common.kv_cache_mem_manager.fp8_static_per_tensor_quant_mem_manager import (
    FP8StaticPerTensorQuantMemManager,
)
from lightllm.common.kv_cache_mem_manager.ppl_int4kv_mem_manager import PPLINT4KVMemoryManager
from lightllm.common.kv_cache_mem_manager.ppl_int8kv_mem_manager import PPLINT8KVMemoryManager
from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import (
    Qwen3NextMemManager,
    FP8StaticPerHeadQuantQwen3NextMemManager,
    FP8StaticPerTensorQuantQwen3NextMemManager,
)
from lightllm.models.deepseek2 import model as deepseek_module
from lightllm.models.llama import model as llama_module
from lightllm.models.mixtral import model as mixtral_module
from lightllm.models.qwen2 import model as qwen_module
from lightllm.models.qwen3_moe.model import Qwen3MOEModel
from lightllm.models.qwen3next import model as next_module


class KVAllocationReached(Exception):
    """Stop after profiling, before allocator, CUDA buffers or KV scale loading."""


@pytest.fixture
def profile_without_allocating(monkeypatch):
    budget = SimpleNamespace(free=40 * 1024 ** 3, total=80 * 1024 ** 3)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(memory_module.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(memory_module, "get_available_gpu_memory", lambda _world: budget.free / 1024 ** 3)
    monkeypatch.setattr(memory_module, "get_total_gpu_memory", lambda: budget.total / 1024 ** 3)
    monkeypatch.setattr(memory_module, "get_env_start_args", lambda: SimpleNamespace(page_size=1))
    monkeypatch.setattr(memory_module.MemoryManager, "get_pd_kv_move_buffer_size", lambda self: 0)

    def stop_allocation(size):
        raise KVAllocationReached(size)

    monkeypatch.setattr(memory_module, "KvCacheAllocator", stop_allocation)
    return budget


def make_expert(layer=0, full=True, redundant=0):
    rows = 64 + redundant
    expert = FusedMoeWeight.__new__(FusedMoeWeight)
    expert.enable_ep_moe = True
    expert.layer_num_ = layer
    expert.expert_parallel_state = ExpertParallelState(
        512,
        8,
        EPLBState(
            redundant,
            torch.empty((8, redundant), dtype=torch.int64),
            None,
            None,
            torch.empty((40, 512), dtype=torch.int64, device="meta"),
            full_layout=full,
        ),
    )
    # Eight layers fit in an 80 GiB device; no actual large allocation is made.
    expert.w13 = SimpleNamespace(weight=torch.empty((rows, 6144, 4096), dtype=torch.bfloat16, device="meta"))
    expert.w2 = SimpleNamespace(weight=torch.empty((rows, 4096, 3072), dtype=torch.bfloat16, device="meta"))
    return expert


def make_model(model_class=Qwen3MOEModel, full=True, redundant=0):
    model = model_class.__new__(model_class)
    model.args = SimpleNamespace(enable_prefill_eplb=True)
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
    model.trans_layers_weight = [SimpleNamespace(experts=make_expert(layer, full, redundant)) for layer in range(8)]
    return model


@pytest.mark.parametrize("full,redundant,staging", [(True, 0, 2304), (True, 2, 2376), (False, 2, 1152)])
@pytest.mark.parametrize(
    "model_class,module,manager_class",
    [
        (base_module.TpPartBaseModel, base_module, memory_module.MemoryManager),
        (llama_module.LlamaTpPartModel, llama_module, memory_module.MemoryManager),
        (Qwen3MOEModel, qwen_module, memory_module.MemoryManager),
        (deepseek_module.Deepseek2TpPartModel, deepseek_module, Deepseek2MemoryManager),
        (deepseek_module.Deepseek2TpPartModel, deepseek_module, Deepseek3_2MemoryManager),
        (mixtral_module.MixtralTpPartModel, mixtral_module, memory_module.MemoryManager),
        (next_module.Qwen3NextTpPartModel, next_module, Qwen3NextMemManager),
    ],
    ids=["base", "llama", "qwen3", "deepseek2", "deepseek3_2", "mixtral", "qwen3next"],
)
def test_model_profile_reserves_eplb_before_kv_alloc(
    monkeypatch, profile_without_allocating, model_class, module, manager_class, full, redundant, staging
):
    model = make_model(model_class, full, redundant)
    manager = manager_class.__new__(manager_class)

    def construct(*args, **kwargs):
        manager_class.__init__(manager, *args, **kwargs)

    if module is mixtral_module:
        monkeypatch.setattr(module, "MemoryManager", construct)
    elif module is next_module:
        monkeypatch.setattr(module, "Qwen3NextMemManager", construct)
        monkeypatch.setattr(
            module,
            "get_env_start_args",
            lambda: SimpleNamespace(llm_kv_type="None", linear_att_ssm_data_type="bfloat16"),
        )
    else:
        monkeypatch.setattr(module, "select_mem_manager_class", lambda: construct)
    monkeypatch.setattr(module, "get_added_mtp_kv_layer_num", lambda: 0, raising=False)

    with pytest.raises(KVAllocationReached):
        model._init_mem_manager()

    expected = {"eplb_staging": staging * 1024 ** 2, "eplb_sampling": 2 * 8 * 40 * 512 * 8 + 512}
    assert manager.memory_reservations == expected
    budget = profile_without_allocating
    available = budget.free - budget.total * (1 - model.mem_fraction)
    assert manager.size == int((available - sum(expected.values())) / manager.get_cell_size())
    # The original full R0 bug leaves ~1.6 GiB for a 2.25 GiB staging allocation.
    assert budget.free - manager.size * manager.get_cell_size() >= sum(expected.values())


@pytest.mark.parametrize("enabled,draft", [(False, False), (True, True)])
def test_disabled_eplb_and_draft_do_not_reserve_target_buffers(enabled, draft):
    model = make_model()
    model.args.enable_prefill_eplb = enabled
    model.is_mtp_draft_model = draft
    assert model._get_post_profile_memory_reservations() == {}


def test_weight_aliases_are_deduplicated_and_non_eplb_layers_are_excluded():
    model = make_model()
    first, last = model.trans_layers_weight[0].experts, model.trans_layers_weight[-1].experts
    ordinary = make_expert()
    ordinary.expert_parallel_state = ExpertParallelState(512, 8)
    model.trans_layers_weight = [
        SimpleNamespace(a=last, b=first, alias=first, ordinary=ordinary, dense=object()),
        SimpleNamespace(experts=last),
    ]
    assert model._get_eplb_weights() == [first, last]
    assert model._get_post_profile_memory_reservations()["eplb_sampling"] == 2 * 2 * 40 * 512 * 8 + 512


@pytest.mark.parametrize("redundant,expected", [(0, 27), (2, 45)])
def test_full_staging_includes_odd_rows_and_scalar_quantization_scales(redundant, expected):
    expert = make_expert(redundant=redundant)
    expert.expert_parallel_state = ExpertParallelState(24, 8, expert.expert_parallel_state.eplb)
    rows = 3 + redundant
    expert.w13 = SimpleNamespace(
        weight=torch.empty((rows, 5), dtype=torch.uint8), weight_scale=torch.empty((rows, 1), dtype=torch.float32)
    )
    expert.w2 = SimpleNamespace(weight=torch.empty((rows, 3), dtype=torch.uint8))
    assert eplb_utils.get_eplb_staging_nbytes([expert] * 12) == expected


@pytest.mark.parametrize(
    "manager_class",
    [
        memory_module.MemoryManager,
        Deepseek2MemoryManager,
        Deepseek3_2MemoryManager,
        FP8PerTokenGroupQuantDeepseek3_2MemoryManager,
        PPLINT4KVMemoryManager,
        PPLINT8KVMemoryManager,
        FP8StaticPerHeadQuantMemManager,
        FP8StaticPerTensorQuantMemManager,
        Qwen3NextMemManager,
        FP8StaticPerHeadQuantQwen3NextMemManager,
        FP8StaticPerTensorQuantQwen3NextMemManager,
    ],
)
def test_kv_backend_constructors_deduct_reservations_once(profile_without_allocating, manager_class):
    budget = profile_without_allocating
    budget.free, budget.total = 64 * 1024 ** 2, 0
    reservations = {"eplb_staging": 1024, "eplb_sampling": 512}
    manager = manager_class.__new__(manager_class)
    kwargs = dict(size=None, dtype=torch.bfloat16, head_num=1, head_dim=576, layer_num=8)
    if issubclass(manager_class, Qwen3NextMemManager):
        kwargs["num_kv_heads"] = kwargs.pop("head_num")
        kwargs["full_att_layer_num"] = kwargs.pop("layer_num")
        kwargs["linear_config"] = SimpleNamespace()
    with pytest.raises(KVAllocationReached):
        manager_class.__init__(manager, **kwargs, memory_reservations=reservations)
    assert manager.memory_reservations == reservations
    assert manager.size == (budget.free - 1536) // manager.get_cell_size()
