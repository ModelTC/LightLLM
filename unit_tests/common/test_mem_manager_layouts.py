from types import SimpleNamespace

import pytest
import torch

from lightllm.common.kv_cache_mem_manager import (
    MemoryManager,
    FP8StaticPerHeadQuantMemManager,
    FP8StaticPerTensorQuantMemManager,
    PPLINT4KVMemoryManager,
    PPLINT8KVMemoryManager,
)


def test_layer_mapping_validates_storage_bounds():
    mem = MemoryManager.__new__(MemoryManager)
    mem.layer_num = 3
    assert mem.get_kv_layer_index(2) == 2
    for mapping in ([], [0, -1], [0, 3]):
        with pytest.raises(ValueError, match="allocated layers"):
            mem.set_kv_layer_mapping(mapping)
    mem.set_kv_layer_mapping([0, 0, 1, 1, 2])
    assert [mem.get_kv_layer_index(i) for i in (1, 3, 4)] == [0, 1, 2]


@pytest.mark.parametrize("manager_class", [PPLINT4KVMemoryManager, PPLINT8KVMemoryManager])
def test_integer_cache_capacity_matches_allocated_buffers(monkeypatch, manager_class):
    allocate = torch.empty
    monkeypatch.setattr(torch, "empty", lambda shape, **kw: allocate(shape, dtype=kw["dtype"]))
    mem = manager_class.__new__(manager_class)
    mem.page_size, mem.head_num, mem.head_dim, mem.layer_num = 4, 2, 16, 3
    mem.group_quant_size, mem.kv_dtype, mem.dtype = 8, torch.int8, torch.bfloat16
    mem._init_buffers(8, mem.dtype, mem.head_num, mem.head_dim, mem.layer_num)
    storage_bytes = sum(t.numel() * t.element_size() for t in (mem.kv_buffer, mem.scale_buffer))
    assert storage_bytes == mem.get_cell_size() * 12
    mem.kv_buffer.random_(-128, 128)
    mem.scale_buffer.normal_()
    saved = mem.get_index_kv_buffer(slice(1, 3))
    mem.load_index_kv_buffer(slice(5, 7), saved)
    for name, expected in saved.items():
        assert torch.equal(getattr(mem, name)[:, 5:7], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "manager_class", [MemoryManager, FP8StaticPerHeadQuantMemManager, FP8StaticPerTensorQuantMemManager]
)
def test_mapped_target_and_draft_kv_writes(manager_class):
    mem = manager_class.__new__(manager_class)
    mem.layer_num, mem.head_num = 3, 1
    quantized = manager_class is not MemoryManager
    mem.kv_buffer = torch.zeros((3, 8, 2, 16), dtype=torch.uint8 if quantized else torch.bfloat16, device="cuda")
    mem.scales = torch.tensor([[1, 1], [2, 4], [4, 2]], dtype=torch.float32, device="cuda")
    mem.set_kv_layer_mapping([0, 0, 1, 1, 2])
    operator = mem.operator_class(mem)
    indexes = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
    kv = torch.arange(64, dtype=torch.bfloat16, device="cuda").reshape(2, 2, 16)
    for logical_layer, physical_layer in ((3, 1), (4, 2)):
        operator.copy_kv_to_mem_manager(logical_layer, indexes, kv)
        k, v = mem.get_att_input_params(logical_layer)
        actual = torch.cat((k[indexes], v[indexes]), dim=1)
        expected = kv
        if quantized:
            actual = actual.view(torch.float8_e4m3fn).float()
            expected = (kv.float() / mem.scales[physical_layer].view(1, 2, 1)).to(torch.float8_e4m3fn).float()
        assert torch.equal(actual, expected)
    assert not mem.kv_buffer[0].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("manager_class", [PPLINT4KVMemoryManager, PPLINT8KVMemoryManager])
def test_integer_cache_copy_includes_scales(manager_class):
    mem = manager_class.__new__(manager_class)
    mem.kv_buffer = torch.randint(-128, 128, (2, 8, 4, 16), dtype=torch.int8, device="cuda")
    mem.scale_buffer = torch.randn((2, 8, 4, 2), dtype=torch.bfloat16, device="cuda")
    expected = mem.get_index_kv_buffer([0, 2])
    mem.operator_class(mem).copy_mem_to_mem(torch.tensor([0, 2]), torch.tensor([4, 6]))
    for name, values in expected.items():
        assert torch.equal(getattr(mem, name)[:, [4, 6]], values)


@pytest.mark.parametrize("quantized", [False, True])
def test_models_select_mla_layout_without_storage_inspecting_models(monkeypatch, quantized):
    from lightllm.models.deepseek2.model import Deepseek2TpPartModel
    from lightllm.models.deepseek3_2.model import Deepseek3_2TpPartModel
    from lightllm.common.kv_cache_mem_manager import (
        MlaMemoryManager,
        IndexedMlaMemoryManager,
        FP8IndexedMlaMemoryManager,
    )

    monkeypatch.setattr(
        "lightllm.utils.envs_utils.get_env_start_args",
        lambda: SimpleNamespace(llm_kv_type="fp8kv_dsa" if quantized else "None"),
    )
    assert Deepseek2TpPartModel.get_mem_manager_class() is MlaMemoryManager
    assert Deepseek3_2TpPartModel.get_mem_manager_class() is (
        FP8IndexedMlaMemoryManager if quantized else IndexedMlaMemoryManager
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("global_heads,prefill_tp,decode_tp", [(4, 1, 2), (4, 2, 1), (1, 1, 4), (1, 4, 1)])
def test_kv_transfer_view_roundtrip(global_heads, prefill_tp, decode_tp):
    from lightllm.common.kv_transfer_resources import KVTransferView
    from lightllm.common.kv_page_transfer import KVPageTransfer

    expected = torch.randn((2, 3, 2, global_heads, 16), dtype=torch.bfloat16, device="cuda")

    def views(tp, initialize):
        result = []
        for rank in range(tp):
            local = max(1, global_heads // tp)
            first = rank // max(1, tp // global_heads) * local
            buffer = torch.zeros((2, 8, 2 * local, 16), dtype=torch.bfloat16, device="cuda")
            if initialize:
                buffer[:, [1, 3, 6]] = expected[:, :, :, first : first + local].reshape(2, 3, 2 * local, 16)
            result.append(KVTransferView(buffer, (2, 2 * global_heads, 16), torch.bfloat16, False))
        return result

    sources, destinations = views(prefill_tp, True), views(decode_tp, False)
    writer, reader = KVPageTransfer(sources[0]), KVPageTransfer(destinations[0])
    writer.alloc_paged_kv_move_buffer(1, 8)
    reader.alloc_paged_kv_move_buffer(1, 8)
    writer.write_mem_to_page_kv_move_buffer([1, 3, 6], 0, 0, sources, prefill_tp)
    reader.kv_move_buffer.copy_(writer.kv_move_buffer)
    reader.read_page_kv_move_buffer_to_mem([0, 2, 5], 0, 0, destinations, decode_tp)
    for rank, view in enumerate(destinations):
        local = max(1, global_heads // decode_tp)
        first = rank // max(1, decode_tp // global_heads) * local
        assert torch.equal(
            view.kv_buffer[:, [0, 2, 5]], expected[:, :, :, first : first + local].reshape(2, 3, 2 * local, 16)
        )
        assert not view.kv_buffer[:, [1, 3, 4, 6, 7]].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("indexed", [False, True])
def test_mla_writer_respects_the_storage_view(indexed):
    from lightllm.common.kv_cache_mem_manager import MlaMemoryManager, IndexedMlaMemoryManager

    cls = IndexedMlaMemoryManager if indexed else MlaMemoryManager
    mem = cls.__new__(cls)
    mem.head_dim = 576 + (72 if indexed else 0)
    mem.layer_num = 2
    mem.kv_buffer = torch.full((2, 8, 1, mem.head_dim), 13, dtype=torch.bfloat16, device="cuda")
    mem.set_kv_layer_mapping([0, 0, 1])
    kv = torch.randn((3, 1, 576), dtype=torch.bfloat16, device="cuda")
    indexes = torch.tensor([1, 3, 6], dtype=torch.int32, device="cuda")
    mem.operator_class(mem).copy_kv_to_mem_manager(2, indexes, kv)
    assert torch.equal(mem.get_att_input_params(2)[indexes], kv)
    assert (mem.kv_buffer[0] == 13).all()
    if indexed:
        assert (mem.kv_buffer[..., 576:] == 13).all()


@pytest.mark.parametrize(
    "kv_type,expected",
    [
        ("None", MemoryManager),
        ("fp8kv_sph", FP8StaticPerHeadQuantMemManager),
        ("fp8kv_spt", FP8StaticPerTensorQuantMemManager),
    ],
)
def test_qwen_storage_selection(monkeypatch, kv_type, expected):
    from lightllm.models.qwen3next import model

    monkeypatch.setattr(model, "get_env_start_args", lambda: SimpleNamespace(llm_kv_type=kv_type))
    assert model.Qwen3NextTpPartModel.get_mem_manager_class() is expected
    for unsupported in ("int4kv", "int8kv"):
        monkeypatch.setattr(model, "get_env_start_args", lambda: SimpleNamespace(llm_kv_type=unsupported))
        with pytest.raises(ValueError, match="does not support"):
            model.Qwen3NextTpPartModel.get_mem_manager_class()


def test_glm_storage_selection(monkeypatch):
    from lightllm.models.glm5_next import model
    from lightllm.common.kv_cache_mem_manager import Glm5NextMemManager

    monkeypatch.setattr(model, "get_env_start_args", lambda: SimpleNamespace(llm_kv_type="None"))
    assert model.Glm5NextTpPartModel.get_mem_manager_class() is Glm5NextMemManager
    monkeypatch.setattr(model, "get_env_start_args", lambda: SimpleNamespace(llm_kv_type="fp8kv_sph"))
    with pytest.raises(ValueError, match="does not support"):
        model.Glm5NextTpPartModel.get_mem_manager_class()


@pytest.mark.parametrize("manager_class", [FP8StaticPerTensorQuantMemManager, PPLINT8KVMemoryManager])
def test_dp_fetch_startup_rejects_unsupported_storage(monkeypatch, manager_class):
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.server.router.model_infer.mode_backend import base_backend

    mem = manager_class.__new__(manager_class)
    mem.operator = mem.operator_class(mem)
    model = SimpleNamespace(mem_manager=mem)
    backend = base_backend.ModeBackend.__new__(base_backend.ModeBackend)
    monkeypatch.setattr(backend, "init_rank_infos", lambda: None)
    monkeypatch.setattr(base_backend, "init_distributed_env", lambda _: None)
    monkeypatch.setattr(base_backend.dist_group_manager, "create_groups", lambda group_size: None)
    monkeypatch.setattr(base_backend, "TokenLoad", lambda *args: None)
    monkeypatch.setattr(base_backend.PretrainedConfig, "get_config_dict", lambda _: ({}, {}))
    monkeypatch.setattr(base_backend, "get_model", lambda *args: (model, False))
    with pytest.raises(ValueError, match=f"--enable_dp_prompt_cache_fetch.*{manager_class.__name__}"):
        backend.init_model(
            {
                "args": StartArgs(tp=2, dp=2, enable_dp_prompt_cache_fetch=True),
                "world_size": 2,
                "load_way": "HF",
                "weight_dir": "/model",
                "max_total_token_num": 16,
            }
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_glm_dp_fetch_copies_packed_indexer_bytes():
    from lightllm.common.kv_cache_mem_manager import Glm5NextMemManager
    from lightllm.common.kv_transfer_resources import KVTransferView

    sources = [
        torch.randint(0, 256, (2, 8, 1, 1168), dtype=torch.uint8, device="cuda").view(torch.bfloat16) for _ in range(4)
    ]
    mem = Glm5NextMemManager.__new__(Glm5NextMemManager)
    mem.kv_buffer = torch.zeros_like(sources[0])
    operator = mem.operator_class(mem)
    views = [KVTransferView(source, None, source.dtype, True) for source in sources]
    operator.copy_kv_from_other_dp_ranks(
        kv_views=views,
        move_token_indexes=torch.tensor([1, 3, 6], dtype=torch.int64, device="cuda"),
        token_dp_indexes=torch.tensor([0, 1, 1], dtype=torch.int32, device="cuda"),
        mem_indexes=torch.tensor([2, 4, 7], dtype=torch.int64, device="cuda"),
        dp_size_in_node=2,
        rank_in_dp=1,
    )
    expected = torch.stack([sources[1][:, 1], sources[3][:, 3], sources[3][:, 6]], dim=1)
    assert torch.equal(mem.kv_buffer.view(torch.uint8)[:, [2, 4, 7]], expected.view(torch.uint8))
    assert not mem.kv_buffer.view(torch.uint8)[:, [0, 1, 3, 5, 6]].any()


@pytest.mark.parametrize("layer_count", [5, 8])
@pytest.mark.parametrize("draft_layers", [0, 2])
def test_hybrid_mapping_excludes_linear_layers(layer_count, draft_layers):
    from lightllm.common.state_cache_manager import LinearAttCacheConfig

    config = LinearAttCacheConfig(
        tp_world_size=1,
        full_att_all_num_kv_heads=1,
        full_att_dtype=torch.bfloat16,
        full_att_num_kv_heads=1,
        full_att_head_dim=8,
        global_linear_k_heads=1,
        global_linear_v_heads=1,
        num_linear_k_heads=1,
        num_linear_v_heads=1,
        head_linear_k_dim=8,
        head_linear_v_dim=8,
        conv_kernel_size=4,
        linear_layer_num=layer_count - layer_count // 4,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
        full_attention_interval=4,
        all_layer_num=layer_count,
        draft_full_att_kv_layer_num=draft_layers,
    )
    expected = (None, None, None, 0, None)
    if layer_count == 8:
        expected += (None, None, 1)
    expected += tuple(range(layer_count // 4, layer_count // 4 + draft_layers))
    assert config.get_full_att_kv_layer_mapping() == expected
    mem = MemoryManager.__new__(MemoryManager)
    mem.layer_num, mem.head_num = config.get_full_att_kv_layer_num_with_draft_model(), 1
    mem.kv_buffer = torch.randn(mem.layer_num, 2, 2, 8)
    mem.set_kv_layer_mapping(config.get_full_att_kv_layer_mapping())
    for layer, slot in enumerate(expected):
        if slot is None:
            with pytest.raises(ValueError, match="no full-attention KV storage"):
                config.get_full_att_kv_layer_index(layer)
            with pytest.raises(ValueError, match="no KV storage"):
                mem.get_att_input_params(layer)
            with pytest.raises(ValueError, match="no KV storage"):
                mem.operator_class(mem).copy_kv_to_mem_manager(layer, torch.tensor([0]), torch.empty(1, 2, 8))
        else:
            assert config.get_full_att_kv_layer_index(layer) == slot
            k, v = mem.get_att_input_params(layer)
            torch.testing.assert_close(torch.cat((k, v), dim=1), mem.kv_buffer[slot])
