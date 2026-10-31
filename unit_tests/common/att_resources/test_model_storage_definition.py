"""Model-specific storage geometry must also describe external cache pages."""

from __future__ import annotations

from unit_tests.common.att_resources.model_helpers import read_kv_storage

from unit_tests.common.att_resources.model_helpers import rebuild_full_state

from typing import TYPE_CHECKING
import dataclasses
import json
from functools import partial

from unit_tests.common.att_resources.model_helpers import prepare_model_states

import pytest
import torch

from lightllm.models.bloom.model import BloomTpPartModel
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen2.model import Qwen2TpPartModel
from lightllm.models.starcoder.model import StarcoderTpPartModel
from lightllm.models.starcoder2.model import Starcoder2TpPartModel
from lightllm.models.mistral.model import MistralTpPartModel
from lightllm.models.mixtral.model import MixtralTpPartModel
from lightllm.models.gemma_2b.model import Gemma_2bTpPartModel
from lightllm.models.gemma3.model import Gemma3TpPartModel
from lightllm.models.gemma4.model import Gemma4TpPartModel
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.cpu_cache_layout import build_cpu_cache_layouts
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.allocators import FullAttKVAllocator
from unit_tests.common.att_resources.test_quantized_full_state import KINDS, make_quant_state

if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from pathlib import Path


@pytest.fixture
def args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    value = StartArgs(
        model_dir=str(tmp_path),
        data_type="float16",
        tp=4,
        dp=1,
        page_size=4,
        enable_cpu_cache=True,
        cpu_cache_token_page_size=8,
        run_mode="prefill",
        pd_kv_page_size=8,
    )
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(dataclasses.asdict(value)))
    get_env_start_args.cache_clear()
    yield value
    get_env_start_args.cache_clear()


CASES = [
    (BloomTpPartModel, {}, 8, 16, 2),
    (LlamaTpPartModel, {"num_key_value_heads": 4, "head_dim": 32}, 4, 32, 1),
    (Qwen2TpPartModel, {"num_key_value_heads": 2}, 2, 16, 1),
    (StarcoderTpPartModel, {}, 1, 16, 1),
    (Starcoder2TpPartModel, {"num_key_value_heads": 4}, 4, 16, 1),
    (MistralTpPartModel, {"num_key_value_heads": 4, "head_dim": 32}, 4, 32, 1),
    (MixtralTpPartModel, {"num_key_value_heads": 4}, 4, 16, 1),
    (Gemma_2bTpPartModel, {"num_key_value_heads": 2}, 2, 16, 2),
    (Gemma3TpPartModel, {"num_key_value_heads": 4}, 4, 256, 1),
    (
        Gemma4TpPartModel,
        {"num_key_value_heads": 4, "head_dim": 256, "num_global_key_value_heads": None, "global_head_dim": 512},
        8,
        256,
        2,
    ),
    (
        Gemma4TpPartModel,
        {"num_key_value_heads": 16, "head_dim": 256, "num_global_key_value_heads": 4, "global_head_dim": 512},
        16,
        256,
        4,
    ),
    (
        Gemma4TpPartModel,
        {"num_key_value_heads": 4, "head_dim": 256, "global_head_dim": 256,
         "num_kv_shared_layers": 1, "layer_types": ["full_attention", "full_attention"]},
        4,
        256,
        1,
    ),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("model_cls,overrides,global_heads,dim,local_heads", CASES)
def test_model_and_cpu_service_use_identical_storage_geometry(
    args: StartArgs | EasyDict,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_cls: type[TpPartBaseModel],
    overrides: dict[str, object],
    global_heads: int,
    dim: int,
    local_heads: int,
):
    # Exercise raw HF aliases as well as a wrapped multimodal config.
    config = dict(
        model_type="llama",
        n_head=8,
        n_embd=128,
        n_layer=2,
        **overrides,
    )
    wrapped = {"model_type": "gemma4", "text_config": config} if model_cls is Gemma4TpPartModel else config
    (tmp_path / "config.json").write_text(json.dumps(wrapped))
    monkeypatch.setattr("lightllm.utils.llm_utils.get_llm_model_class", lambda: model_cls)
    service_layout = next(iter(build_cpu_cache_layouts().values()))
    for rank in range(args.tp):
        model = model_cls.__new__(model_cls)
        model.config, model.args = wrapped, args
        model.tp_rank_, model.tp_world_size_ = rank, args.tp
        model.max_total_token_num, model.max_req_num = 19, 1
        prepare_model_states(model)
        resources = model.att_state_manager
        resources.finish_state_registration()
        resources.initialize_storage(args, token_capacity=model.max_total_token_num, mem_fraction=args.mem_fraction)
        (state,) = resources.get_full_attention_states("main")
        assert state.config.head_num == local_heads and state.config.head_dim == dim
        storage_layers = 2 - (config.get("num_kv_shared_layers") or 0)
        assert state.k_storage.shape == state.v_storage.shape == (storage_layers, 20, local_heads, dim)
        if storage_layers == 1:
            assert state.config.logical_layer_groups == ((0, 1),)
            assert resources.get_state("main", "gqa", 0) is resources.get_state("main", "gqa", 1)
            assert state.get_layer_view(logical_layer_id=0)[0].data_ptr() == state.get_layer_view(logical_layer_id=1)[0].data_ptr()
        assert read_kv_storage(state).dtype == (torch.bfloat16 if model_cls is Gemma3TpPartModel else torch.float16)
        assert resources.full_att_allocator.size == 16
        layout = resources.get_cpu_page_layouts()[state.state_name]
        assert (layout.byte_start, layout.byte_end) == (service_layout.byte_start, service_layout.byte_end)
        # 按物理层数计算传输页；共享消费者不重复增加 KV 或传输容量。
        expected_pd_bytes = storage_layers * args.pd_kv_page_size * 2 * global_heads * dim * read_kv_storage(state).element_size()
        assert resources.get_pd_page_bytes() == expected_pd_bytes
        fields = layout.regions
        assert all(region.shape[2] == global_heads for region in fields)
        assert not hasattr(model, "mem_manager")


@pytest.mark.parametrize("kind", KINDS)
def test_gpu_profile_counts_token_fields_hold_and_transfer_reserve(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    pd_args: EasyDict,
):
    state = make_quant_state(kind, "cpu")
    reference = AttentionStateManager(request_capacity=1)
    reference.register_state("main", state)
    reference.full_att_allocator = FullAttKVAllocator(state.capacity, state.page_size, pin_memory=False)
    reference.finish_state_registration()
    args = StartArgs(
        page_size=4,
        run_mode="prefill",
        pd_kv_page_size=8,
        pd_kv_page_num=3,
    )
    reserve = reference.get_pd_page_bytes() * args.pd_kv_page_num
    required = state.get_token_bytes() * (32 + args.page_size)
    # One byte below the full 32-token budget must drop a complete page.
    total = 1 << 30
    free = required - 1 + reserve + total // 4
    observed_devices = []

    def memory_info(device: str | torch.device | int):
        observed_devices.append(device)
        return free, total

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", memory_info)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr("lightllm.common.attention_state.manager.get_current_device_id", lambda: 3)
    monkeypatch.setattr(
        "lightllm.common.attention_state.manager.FullAttKVAllocator", partial(FullAttKVAllocator, pin_memory=False)
    )
    template = rebuild_full_state(
        template=state,
        config=state.config,
        token_capacity=0,
        device="meta",
    )
    resources = AttentionStateManager(request_capacity=1)
    resources.register_state("main", rebuild_full_state(template, config=template.config))
    resources.finish_state_registration()
    resources.initialize_storage(
        args=args,
        token_capacity=None,
        mem_fraction=0.75,
        device="cpu",
    )
    assert resources.full_att_allocator.size == 28
    assert resources.get_pd_page_bytes() == reference.get_pd_page_bytes()
    assert observed_devices == [torch.device("cuda", 3)]


def test_cpu_layout_uses_all_model_factory_states(
    args: StartArgs | EasyDict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """同一模型返回多种方案和共享层时，CPU 页采用实际 State 列表。"""
    from lightllm.common.attention_state.full_att import FullAttentionState, GQAFullAttentionState

    class MixedStorageModel(LlamaTpPartModel):
        @classmethod
        def create_full_att_states(
            cls,
            model_config: dict[str, object],
            args: StartArgs | EasyDict,
            model_logical_layer_ids: tuple[int, ...],
            tp_rank: int = 0,
            tp_world_size: int = 1,
        ) -> list[FullAttentionState]:
            (template,) = super().create_full_att_states(
                model_config=model_config,
                args=args,
                model_logical_layer_ids=model_logical_layer_ids,
                tp_rank=tp_rank,
                tp_world_size=tp_world_size,
            )
            return [
                GQAFullAttentionState(
                    layout_config=dataclasses.replace(
                        template.config, logical_layer_groups=(model_logical_layer_ids[:2],), dtype=torch.bfloat16,
                    ), page_size=args.page_size, device="meta",
                ),
                GQAFullAttentionState(
                    layout_config=dataclasses.replace(
                        template.config, logical_layer_groups=(model_logical_layer_ids[2:],), dtype=torch.float16,
                    ), page_size=args.page_size, device="meta",
                ),
            ]

    config = dict(
        model_type="llama",
        num_attention_heads=8,
        hidden_size=128,
        num_hidden_layers=3,
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr("lightllm.utils.llm_utils.get_llm_model_class", lambda: MixedStorageModel)
    cpu_layouts = build_cpu_cache_layouts()
    model = MixedStorageModel.__new__(MixedStorageModel)
    model.config, model.args = config, args
    model.tp_rank_, model.tp_world_size_, model.max_req_num = 0, 1, 1
    prepare_model_states(model)
    model.att_state_manager.finish_state_registration()
    runtime_layouts = model.att_state_manager.get_cpu_page_layouts()
    assert set(cpu_layouts) == set(runtime_layouts) == {"main_gqa_bf16_state", "main_gqa_fp16_state"}
    assert cpu_layouts == runtime_layouts
