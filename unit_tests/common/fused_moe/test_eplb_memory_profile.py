"""EPLB buffers reduce the existing mem_fraction before KV profiling."""
import json
from types import SimpleNamespace
from contextlib import nullcontext

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
from lightllm.common.kv_cache_mem_manager.glm5_next_mem_manager import Glm5NextMemManager
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
from lightllm.models.deepseek_mtp.model import Deepseek3MTPModel
from lightllm.models.glm5_next import model as glm_module
from lightllm.models.glm5_next_mtp.model import Glm5NextMTPModel
from lightllm.models.llama import model as llama_module
from lightllm.models.mixtral import model as mixtral_module
from lightllm.models.qwen2 import model as qwen_module
from lightllm.models.qwen3_moe.model import Qwen3MOEModel
from lightllm.models.qwen3_moe_mtp.model import Qwen3MOEMTPModel
from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
from lightllm.models.qwen3_5_mtp.model import Qwen3_5MTPModel
from lightllm.models.qwen3next import model as next_module
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils import device_utils, profile_max_tokens


@pytest.fixture(autouse=True)
def supported_eplb_device(monkeypatch):
    monkeypatch.setattr(device_utils, "is_sm100_gpu", lambda: False)


class KVAllocationReached(Exception):
    """Stop after profiling, before allocator, CUDA buffers or KV scale loading."""


@pytest.fixture
def profile_without_allocating(monkeypatch):
    budget = SimpleNamespace(free=40 * 1024 ** 3, total=80 * 1024 ** 3)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(total_memory=budget.total))
    monkeypatch.setattr(profile_max_tokens, "get_current_device_id", lambda: 0)
    monkeypatch.setattr(memory_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(memory_module.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(memory_module, "get_available_gpu_memory", lambda _world: budget.free / 1024 ** 3)
    monkeypatch.setattr(memory_module, "get_total_gpu_memory", lambda: budget.total / 1024 ** 3)
    monkeypatch.setattr(memory_module, "get_env_start_args", lambda: SimpleNamespace(page_size=1))
    monkeypatch.setattr(memory_module.MemoryManager, "get_pd_kv_move_buffer_size", lambda self: 0)

    def stop_allocation(size):
        raise KVAllocationReached(size)

    monkeypatch.setattr(memory_module, "KvCacheAllocator", stop_allocation)
    return budget


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


@pytest.mark.parametrize("redundant,staging", [(0, 4608), (2, 4752)])
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
        (glm_module.Glm5NextTpPartModel, glm_module, Glm5NextMemManager),
    ],
    ids=["base", "llama", "qwen3", "deepseek2", "deepseek3_2", "mixtral", "qwen3next", "glm5next"],
)
def test_model_profile_reserves_eplb_before_kv_alloc(
    monkeypatch, profile_without_allocating, model_class, module, manager_class, redundant, staging
):
    model = make_model(model_class, redundant)
    manager = manager_class.__new__(manager_class)

    def construct(*args, **kwargs):
        manager_class.__init__(manager, *args, **kwargs)

    if module is mixtral_module:
        monkeypatch.setattr(module, "MemoryManager", construct)
    elif module is glm_module:
        model.linear_config = SimpleNamespace(
            full_att_head_dim=644, get_full_att_kv_layer_num_with_draft_model=lambda: 8
        )
        monkeypatch.setattr(module, "Glm5NextMemManager", construct)
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

    budget = profile_without_allocating
    original_fraction = model.mem_fraction
    with profile_max_tokens.profile_model_memory(model):
        pass
    with pytest.raises(KVAllocationReached):
        model._init_mem_manager()

    reserved_bytes = staging * 1024 ** 2 + 2 * 8 * 40 * 512 * 8 + 512
    assert model.mem_fraction == pytest.approx(original_fraction - reserved_bytes / budget.total)
    available = budget.free - budget.total * (1 - model.mem_fraction)
    assert manager.size == int(available / manager.get_cell_size())
    # The KV allocation must leave space for the complete staging layer.
    assert budget.free - manager.size * manager.get_cell_size() >= reserved_bytes


@pytest.mark.parametrize(
    "enabled,draft,size,run_mode",
    [
        (False, False, None, "prefill"),
        (True, True, None, "prefill"),
        (True, False, 1024, "prefill"),
        (True, False, None, "decode"),
    ],
)
def test_inactive_eplb_draft_and_explicit_kv_size_keep_mem_fraction(monkeypatch, enabled, draft, size, run_mode):
    model = make_model()
    model.args.enable_prefill_eplb = enabled
    model.args.run_mode = run_mode
    model.is_mtp_draft_model = draft
    model.max_total_token_num = size
    monkeypatch.setattr(eplb_utils, "find_ep_moe_weights", lambda _: pytest.fail("EPLB budgeting must be skipped"))
    with profile_max_tokens.profile_model_memory(model):
        pass
    assert model.mem_fraction == 0.98


def test_weight_aliases_are_deduplicated_and_non_eplb_layers_are_excluded(profile_without_allocating):
    model = make_model()
    first, last = model.trans_layers_weight[0].experts, model.trans_layers_weight[-1].experts
    ordinary = make_expert()
    ordinary.expert_parallel_state = ExpertParallelState(512, 8)
    model.trans_layers_weight = [
        SimpleNamespace(a=last, b=first, alias=first, ordinary=ordinary, dense=object()),
        SimpleNamespace(experts=last),
    ]
    assert [
        weight for weight in eplb_utils.find_ep_moe_weights(model) if weight.expert_parallel_state.eplb is not None
    ] == [first, last]
    expected_sampling_bytes = 2 * 2 * 40 * 512 * 8 + 512
    assert eplb_utils.get_eplb_sampling_peak_nbytes([first, last]) == expected_sampling_bytes
    reserved_bytes = eplb_utils.get_eplb_staging_nbytes([first, last]) + expected_sampling_bytes
    assert profile_max_tokens.get_eplb_adjusted_mem_fraction(model) == pytest.approx(
        model.mem_fraction - reserved_bytes / profile_without_allocating.total
    )


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
        Glm5NextMemManager,
        FP8StaticPerHeadQuantQwen3NextMemManager,
        FP8StaticPerTensorQuantQwen3NextMemManager,
    ],
)
def test_kv_backend_constructors_use_adjusted_mem_fraction_once(profile_without_allocating, manager_class):
    budget = profile_without_allocating
    model = make_model()
    with profile_max_tokens.profile_model_memory(model):
        pass
    manager = manager_class.__new__(manager_class)
    kwargs = dict(
        size=None, dtype=torch.bfloat16, head_num=1, head_dim=576, layer_num=8, mem_fraction=model.mem_fraction
    )
    if issubclass(manager_class, Qwen3NextMemManager):
        kwargs["num_kv_heads"] = kwargs.pop("head_num")
        kwargs["full_att_layer_num"] = kwargs.pop("layer_num")
        kwargs["linear_config"] = SimpleNamespace()
    with pytest.raises(KVAllocationReached):
        manager_class.__init__(manager, **kwargs)
    available = budget.free - budget.total * (1 - model.mem_fraction)
    assert manager.size == int(available / manager.get_cell_size())


@pytest.mark.parametrize("eplb", [False, True])
def test_weight_profiling_reserves_mtp_and_optional_eplb(monkeypatch, profile_without_allocating, eplb):
    model = make_model()
    model.args.enable_prefill_eplb = eplb
    model.args.mtp_mode = "vanilla"
    original_fraction = model.mem_fraction
    allocations = iter([2 * 1024 ** 3, 10 * 1024 ** 3])
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: next(allocations))
    monkeypatch.setattr(profile_max_tokens, "get_mtp_weight_layer_num", lambda: 1)
    weights = [layer.experts for layer in model.trans_layers_weight]
    eplb_bytes = (
        eplb_utils.get_eplb_staging_nbytes(weights) + eplb_utils.get_eplb_sampling_peak_nbytes(weights) if eplb else 0
    )
    with profile_max_tokens.profile_model_memory(model):
        assert model.mem_fraction == original_fraction
    assert model.mem_fraction == pytest.approx(
        original_fraction - (1024 ** 3 + eplb_bytes) / profile_without_allocating.total
    )


def test_distributed_budget_uses_minimum_mem_fraction(monkeypatch, profile_without_allocating):
    model = make_model()
    weights = [layer.experts for layer in model.trans_layers_weight]
    reserved_bytes = eplb_utils.get_eplb_staging_nbytes(weights) + eplb_utils.get_eplb_sampling_peak_nbytes(weights)
    expected_local = model.mem_fraction - reserved_bytes / profile_without_allocating.total
    monkeypatch.setattr(profile_max_tokens, "get_current_device_id", lambda: 3)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)
    original_tensor = torch.tensor
    devices = []

    def cpu_tensor(value, *, dtype, device):
        devices.append(device)
        return original_tensor(value, dtype=dtype)

    def all_reduce(fraction, op):
        assert fraction.item() == pytest.approx(expected_local)
        assert op == torch.distributed.ReduceOp.MIN
        fraction.fill_(0.8)  # Another rank has less remaining memory.

    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    assert profile_max_tokens.get_eplb_adjusted_mem_fraction(model) == pytest.approx(0.8)
    assert devices == ["cuda:3"]


def test_eplb_without_experts_keeps_mem_fraction():
    model = make_model()
    model.trans_layers_weight = []
    assert profile_max_tokens.get_eplb_adjusted_mem_fraction(model) == model.mem_fraction


def test_insufficient_budget_is_rejected_before_kv_allocation(profile_without_allocating):
    model = make_model()
    model.mem_fraction = 0.001
    with pytest.raises(AssertionError, match="EPLB buffers leave no memory"):
        profile_max_tokens.get_eplb_adjusted_mem_fraction(model)


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


def test_base_model_passes_adjusted_fraction_to_overridden_kv_constructor(monkeypatch, profile_without_allocating):
    template = make_model()

    class Model(base_module.TpPartBaseModel):
        def _init_config(self):
            self.config = template.config

        def _init_weights(self):
            self.trans_layers_weight = template.trans_layers_weight

        def _init_req_manager(self):
            # Weight profiling has already reserved EPLB memory; request-state
            # allocation must still be visible to KV capacity profiling.
            assert self.mem_fraction < 0.9
            profile_without_allocating.free -= 1024

        def _init_mem_manager(self):
            self.mem_manager = memory_module.MemoryManager.__new__(memory_module.MemoryManager)
            memory_module.MemoryManager.__init__(
                self.mem_manager, None, torch.bfloat16, 8, 128, 8, mem_fraction=self.mem_fraction
            )

    for name in ("_verify_must", "_verify_params", "_init_quant"):
        monkeypatch.setattr(Model, name, lambda self: None)
    monkeypatch.setattr(
        base_module, "get_env_start_args", lambda: StartArgs(enable_ep_moe=True, enable_prefill_eplb=True)
    )
    monkeypatch.setattr(base_module, "get_llm_data_type", lambda: torch.bfloat16)
    monkeypatch.setattr(base_module, "get_dp_world_size", lambda: 1)
    monkeypatch.setattr(
        base_module, "TorchMemorySaverWrapper", lambda enabled: SimpleNamespace(region=lambda **kwargs: nullcontext())
    )
    monkeypatch.setattr(
        base_module.MtpManager,
        "get_instance",
        lambda: SimpleNamespace(
            get_decode_tokens_per_request=lambda draft: 1, get_decode_batch_alignment=lambda draft: 1
        ),
    )
    model = Model.__new__(Model)
    with pytest.raises(KVAllocationReached):
        Model.__init__(model, dict(run_mode="prefill", weight_dir="unused", max_total_token_num=None))
    weights = [layer.experts for layer in model.trans_layers_weight]
    reserved_bytes = eplb_utils.get_eplb_staging_nbytes(weights) + eplb_utils.get_eplb_sampling_peak_nbytes(weights)
    assert model.mem_fraction == pytest.approx(0.9 - reserved_bytes / profile_without_allocating.total)
    available = profile_without_allocating.free - profile_without_allocating.total * (1 - model.mem_fraction)
    assert model.mem_manager.size == int(available / model.mem_manager.get_cell_size())
