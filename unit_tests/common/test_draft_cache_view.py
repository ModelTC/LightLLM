from types import SimpleNamespace

import pytest
import torch

from lightllm.common.kv_cache_mem_manager import mem_manager as mem_module
from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import (
    FP8StaticPerHeadQuantQwen3NextMemManager,
    Qwen3NextMemManager,
)
from lightllm.common.state_cache_manager import LinearAttCacheConfig


def make_manager(tp_size=1, dtype=torch.bfloat16, heads=4, head_dim=256, quant=False, device="cpu"):
    cls = FP8StaticPerHeadQuantQwen3NextMemManager if quant else Qwen3NextMemManager
    manager = cls.__new__(cls)
    manager.head_num = max(heads // tp_size, 1)
    manager.head_dim = head_dim
    manager.layer_num = 3
    manager.page_size = 4
    manager.size = 8
    manager.dtype = dtype
    manager.allocator = object()
    manager.HOLD_TOKEN_MEMINDEXES = (8, 9, 10, 11)
    manager.linear_config = LinearAttCacheConfig(
        tp_world_size=tp_size,
        full_att_all_num_kv_heads=heads,
        full_att_dtype=dtype,
        full_att_num_kv_heads=manager.head_num,
        full_att_head_dim=head_dim,
        global_linear_k_heads=8,
        global_linear_v_heads=8,
        num_linear_k_heads=8 // tp_size,
        num_linear_v_heads=8 // tp_size,
        head_linear_k_dim=8,
        head_linear_v_dim=8,
        conv_kernel_size=4,
        linear_layer_num=6,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
        full_attention_interval=4,
        all_layer_num=8,
        draft_full_att_kv_layer_num=1,
    )
    manager.kv_buffer = torch.zeros((3, 12, 2 * manager.head_num, head_dim), dtype=dtype, device=device)
    manager.operator = manager.operator_class(manager)
    if quant:
        manager.scales = torch.arange(1, 1 + 3 * 2 * manager.head_num, device=device).float().view(3, -1)
        manager.q_scales = torch.arange(1, 1 + 3 * manager.head_num, device=device).float().view(3, -1)
        alternate_heads = 8 // tp_size
        manager._kv_layout_scales = {
            (manager.head_num, head_dim): (manager.scales, manager.q_scales),
            (alternate_heads, 128): (
                torch.arange(1, 1 + 3 * 2 * alternate_heads, device=device).float().view(3, -1),
                torch.arange(1, 1 + 3 * alternate_heads, device=device).float().view(3, -1),
            ),
        }
    return manager


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.uint8])
def test_view_shares_slots_but_preserves_target_layout(tp_size, dtype):
    target = make_manager(tp_size, dtype, quant=dtype == torch.uint8)
    draft = target.get_kv_layout_view(8, 128)
    assert draft is not target
    assert draft.allocator is target.allocator
    assert draft.HOLD_TOKEN_MEMINDEXES is target.HOLD_TOKEN_MEMINDEXES
    assert draft.operator is not target.operator
    assert draft.operator.mem_manager is draft
    assert target.operator.mem_manager is target
    assert (target.head_num, target.head_dim) == (4 // tp_size, 256)
    assert (draft.head_num, draft.head_dim) == (8 // tp_size, 128)
    assert draft.kv_buffer.data_ptr() == target.kv_buffer.data_ptr()
    assert draft.kv_buffer.is_contiguous()
    # The last physical page is reserved for padding; its slots must also alias.
    k, v = draft.get_att_input_params(8)
    k[11].fill_(3)
    v[11].fill_(7)
    target_k, target_v = target.get_att_input_params(8)
    assert torch.all(target_k[11] == 3)
    assert torch.all(target_v[11] == 7)
    assert torch.count_nonzero(target.kv_buffer[:2]) == 0


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_identical_layout_preserves_manager(tp_size):
    target = make_manager(tp_size)
    assert target.get_kv_layout_view(4, 256) is target


@pytest.mark.parametrize("quant", [False, True])
def test_layout_view_can_be_reused_and_converted_back(quant):
    target = make_manager(dtype=torch.uint8 if quant else torch.bfloat16, quant=quant)
    view = target.get_kv_layout_view(8, 128)
    assert view.get_kv_layout_view(8, 128) is view
    restored = view.get_kv_layout_view(4, 256)
    assert restored.kv_buffer.data_ptr() == target.kv_buffer.data_ptr()
    assert restored.kv_buffer.shape == target.kv_buffer.shape
    if quant:
        torch.testing.assert_close(restored.scales, target.scales)
        torch.testing.assert_close(restored.q_scales, target.q_scales)


@pytest.mark.parametrize(
    "tp_size,heads,dim,error",
    [
        (8, 8, 128, "different per-rank KV widths"),
        (1, 8, 256, "equal global KV widths"),
        (3, 8, 128, "cannot be sharded or replicated"),
        (1, 0, 128, "cannot be sharded or replicated"),
        (1, 8, 0, "equal global KV widths"),
    ],
)
def test_unsupported_layout_fails_before_rebinding(tp_size, heads, dim, error):
    target = make_manager(tp_size)
    operator, buffer = target.operator, target.kv_buffer
    with pytest.raises(ValueError, match=error):
        target.get_kv_layout_view(heads, dim)
    assert target.kv_buffer is buffer
    assert target.operator is operator
    assert operator.mem_manager is target


def test_fp8_layout_requires_independent_calibration():
    target = make_manager(dtype=torch.uint8, quant=True)
    del target._kv_layout_scales[(8, 128)]
    with pytest.raises(ValueError, match="No per-head FP8 calibration"):
        target.get_kv_layout_view(8, 128)


def page_io_reference(mem_indexes, page_tensor, kv_buffer, tp_index, tp_world_size, mode):
    local_heads = kv_buffer.shape[2] // 2
    global_heads = page_tensor.shape[2] // 2
    repeat = local_heads * tp_world_size // global_heads
    start = tp_index // repeat * local_heads
    for local_base, global_base in [(0, start), (local_heads, global_heads + start)]:
        for token_idx, mem_idx in enumerate(mem_indexes.tolist()):
            cache = kv_buffer[:, mem_idx, local_base : local_base + local_heads]
            page = page_tensor[token_idx, :, global_base : global_base + local_heads]
            if mode == "write":
                page.copy_(cache)
            else:
                cache.copy_(page)


@pytest.mark.parametrize("source_tp,dest_tp", [(s, d) for s in [1, 2, 4] for d in [1, 2, 4]])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.uint8])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")),
    ],
)
def test_target_owned_pd_transport_roundtrips_draft_across_tp(monkeypatch, source_tp, dest_tp, dtype, device):
    if device == "cpu":
        monkeypatch.setattr(mem_module, "page_io", page_io_reference)
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)
        tensor = torch.tensor
        monkeypatch.setattr(
            torch,
            "tensor",
            lambda *args, **kwargs: tensor(*args, **{k: v for k, v in kwargs.items() if k != "pin_memory"}),
        )
    source = [make_manager(source_tp, dtype, device=device) for _ in range(source_tp)]
    dest = [make_manager(dest_tp, dtype, device=device) for _ in range(dest_tp)]
    page = torch.zeros((1, 2, 3, 8, 256), dtype=dtype, device=device)
    for manager in [source[0], dest[0]]:
        manager.kv_move_buffer = page
        manager._buffer_mem_indexes_tensors = [torch.empty(2, dtype=torch.int64, pin_memory=device == "cuda")]
    expected = torch.arange(2 * 16 * 128, device=device).remainder(127).reshape(2, 16, 128).to(dtype)
    for rank, manager in enumerate(source):
        draft = manager.get_kv_layout_view(8, 128)
        k, v = draft.get_att_input_params(8)
        n = draft.head_num
        k[1:3] = expected[:, rank * n : (rank + 1) * n]
        v[1:3] = expected[:, 8 + rank * n : 8 + (rank + 1) * n]
    source[0].write_mem_to_page_kv_move_buffer([1, 2], 0, 0, source, source_tp)
    dest[0].read_page_kv_move_buffer_to_mem([3, 4], 0, 0, dest, dest_tp)
    for rank, manager in enumerate(dest):
        draft = manager.get_kv_layout_view(8, 128)
        k, v = draft.get_att_input_params(8)
        n = draft.head_num
        torch.testing.assert_close(k[3:5], expected[:, rank * n : (rank + 1) * n])
        torch.testing.assert_close(v[3:5], expected[:, 8 + rank * n : 8 + (rank + 1) * n])
        assert torch.count_nonzero(manager.kv_buffer[:2]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("quant", [False, True])
def test_draft_operator_writes_and_reads_its_own_layout_on_cuda(quant, load_calibration):
    target = load_calibration(device="cuda") if quant else make_manager(device="cuda")
    draft = target.get_kv_layout_view(8, 128, layer_start=2, layer_num=1)
    kv = torch.randn((2, 16, 128), dtype=torch.bfloat16, device="cuda")
    slots = torch.tensor([1, 9], dtype=torch.int32, device="cuda")
    draft.operator.copy_kv_to_mem_manager(8, slots, kv)
    k, v = draft.get_att_input_params(8)
    actual = torch.cat((k[slots.long()], v[slots.long()]), dim=1)
    if quant:
        scales = draft.scales[2].view(1, 16, 1)
        expected = (kv.float() / scales).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        expected = kv
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(target.kv_buffer[:2]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.uint8])
def test_target_owned_cpu_cache_restores_draft_bytes(monkeypatch, tp_size, dtype):
    from lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy import (
        copy_kv_buffer_to_cpu_cache,
        copy_cpu_cache_to_kv_buffer,
    )
    from lightllm.common.state_cache_manager import linear_att

    args = SimpleNamespace(linear_att_page_block_num=1, linear_att_hash_page_size=8, cpu_cache_token_page_size=8)
    monkeypatch.setattr(linear_att, "get_env_start_args", lambda: args)
    manager = make_manager(tp_size, dtype, device="cuda")
    config = manager.linear_config
    cpu_pages = torch.zeros((1, config.get_cpu_cache_big_page_bytes()), dtype=torch.uint8, pin_memory=True)
    conv = torch.zeros((1, 6, *config.get_conv_state_shape()), dtype=torch.bfloat16, pin_memory=True)
    ssm = torch.zeros((1, 6, *config.get_ssm_state_shape()), dtype=torch.float32, pin_memory=True)
    indexes = torch.arange(8, dtype=torch.int32, device="cuda")
    zero = torch.zeros(1, dtype=torch.int64, device="cuda")
    ready = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    common = dict(
        mem_indexes=indexes,
        page_indexes=zero,
        big_page_buffer_ids=zero,
        cpu_kv_conv_state=conv,
        cpu_kv_ssm_state=ssm,
        cpu_cache_tensor=cpu_pages,
        tp_world_size=tp_size,
        big_page_token_num=8,
        linear_config=config,
    )
    expected = []
    for rank in range(tp_size):
        draft = manager.get_kv_layout_view(8, 128)
        draft.kv_buffer[:, :8].random_(0, 127)
        expected.append(draft.kv_buffer[:, :8].clone())
        copy_kv_buffer_to_cpu_cache(page_readies=ready, gpu_kv_full_att_state=manager.kv_buffer, tp_rank=rank, **common)
        torch.cuda.synchronize()
    for rank in range(tp_size):
        manager.kv_buffer.zero_()
        copy_cpu_cache_to_kv_buffer(gpu_full_att_kv_state=manager.kv_buffer, tp_rank=rank, **common)
        torch.cuda.synchronize()
        torch.testing.assert_close(manager.get_kv_layout_view(8, 128).kv_buffer[:, :8], expected[rank])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("quant", [False, True])
def test_noncausal_fa3_draft_attention_and_cuda_graph(quant, load_calibration):
    from lightllm.utils.sgl_utils import flash_attn_with_kvcache
    from lightllm.common.basemodel.triton_kernel.quantization.q_per_head_fp8_quant import (
        q_per_head_static_fp8_quant,
    )

    if flash_attn_with_kvcache is None:
        pytest.skip("FA3 is unavailable")
    target = load_calibration(device="cuda") if quant else make_manager(device="cuda")
    draft = target.get_kv_layout_view(8, 128, layer_start=2, layer_num=1)
    generator = torch.Generator(device="cuda").manual_seed(0)
    kv = torch.randn((4, 16, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
    q = torch.randn((2, 32, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
    slots = torch.arange(4, dtype=torch.int32, device="cuda")
    page_table = slots.view(1, 4)
    seq_lens = torch.tensor([4], dtype=torch.int32, device="cuda")
    cu_q = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    k, v = draft.get_att_input_params(8)
    kwargs = dict(
        page_table=page_table,
        cache_seqlens=seq_lens,
        cu_seqlens_q=cu_q,
        max_seqlen_q=2,
        causal=False,
    )
    if quant:
        k = k.view(torch.float8_e4m3fn)
        v = v.view(torch.float8_e4m3fn)
        kwargs.update(
            q_descale=draft.q_scales[2].view(1, 8),
            k_descale=draft.scales[2, :8].view(1, 8),
            v_descale=draft.scales[2, 8:].view(1, 8),
        )

    def forward():
        draft.operator.copy_kv_to_mem_manager(8, slots, kv)
        query = q
        if quant:
            query = q_per_head_static_fp8_quant(q.reshape(2, 8, -1), draft.q_scales[2]).reshape(2, 32, 128)
        return flash_attn_with_kvcache(q=query, k_cache=k.view(12, 1, 8, 128), v_cache=v.view(12, 1, 8, 128), **kwargs)

    # Warm both the copy kernel and FA3 before capture.
    forward()
    forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward()
    draft.kv_buffer.zero_()
    graph.replay()
    query = q
    if quant:
        query = q_per_head_static_fp8_quant(q.reshape(2, 8, -1), draft.q_scales[2]).reshape(2, 32, 128)
    contiguous_output = flash_attn_with_kvcache(
        q=query, k_cache=k.contiguous().view(12, 1, 8, 128), v_cache=v.contiguous().view(12, 1, 8, 128), **kwargs
    )
    torch.testing.assert_close(output, contiguous_output, atol=0, rtol=0)
    if quant:
        q_ref = q_per_head_static_fp8_quant(q.reshape(2, 8, -1), draft.q_scales[2]).float()
        q_ref = (q_ref * draft.q_scales[2].view(1, 8, 1)).reshape(2, 32, 128)
        k_ref = k[:4].float() * draft.scales[2, :8].view(1, 8, 1)
        v_ref = v[:4].float() * draft.scales[2, 8:].view(1, 8, 1)
    else:
        q_ref, k_ref, v_ref = q.float(), k[:4].float(), v[:4].float()
    expected = (
        torch.nn.functional.scaled_dot_product_attention(
            q_ref.transpose(0, 1).unsqueeze(0),
            k_ref.repeat_interleave(4, dim=1).transpose(0, 1).unsqueeze(0),
            v_ref.repeat_interleave(4, dim=1).transpose(0, 1).unsqueeze(0),
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    # FP8 FA3 also rounds intermediate attention probabilities.
    tolerance = 0.06 if quant else 0.015
    torch.testing.assert_close(output.float(), expected, atol=tolerance, rtol=tolerance)
    assert torch.count_nonzero(target.kv_buffer[:2]) == 0


def calibration_layout(layers, heads, dim, offset):
    kv = (torch.arange(layers * 2 * heads).reshape(layers, 2 * heads) + offset) / 1000
    q = (torch.arange(layers * heads).reshape(layers, heads) + offset + 50) / 1000
    return {
        "num_layers": layers,
        "num_head": heads,
        "head_dim": dim,
        "scales_shape": list(kv.shape),
        "scales": kv.tolist(),
        "q_calibration": {"num_head": heads, "scales_shape": list(q.shape), "scales": q.tolist()},
    }


@pytest.fixture
def load_calibration(tmp_path, monkeypatch):
    import json
    from lightllm.common.kv_cache_mem_manager import fp8_static_per_head_quant_mem_manager as fp8

    (tmp_path / "config.json").write_text(json.dumps({"num_key_value_heads": 4}))
    path = tmp_path / "calibration.json"
    args = SimpleNamespace(kv_quant_calibration_config_path=str(path), model_dir=str(tmp_path))
    monkeypatch.setattr(fp8, "get_env_start_args", lambda: args)

    def load(cfg=None, tp_size=1, rank=0, draft_layers=1, device="cpu"):
        if cfg is None:
            cfg = {
                "num_layers": 3,
                "num_target_layers": 2,
                "num_draft_layers": 1,
                "layouts": [calibration_layout(2, 4, 256, 1), calibration_layout(1, 8, 128, 101)],
            }
        path.write_text(json.dumps(cfg))
        monkeypatch.setattr(fp8, "get_dp_world_size", lambda: tp_size)
        monkeypatch.setattr(fp8, "get_current_rank_in_dp", lambda: rank)
        monkeypatch.setattr(fp8, "get_added_mtp_kv_layer_num", lambda: draft_layers)

        def init_storage(manager, size, dtype, heads, dim, layers, *args):
            storage = make_manager(tp_size, dtype, device=device)
            manager.__dict__.update(storage.__dict__)
            manager.layer_num = layers
            manager.kv_buffer = manager.kv_buffer.new_zeros((layers, *manager.kv_buffer.shape[1:]))
            manager.linear_config.draft_full_att_kv_layer_num = draft_layers
            manager.operator = manager.operator_class(manager)

        # Replace storage allocation only; exercise the actual FP8 constructor,
        # JSON parsing, metadata validation and TP scale slicing.
        monkeypatch.setattr(mem_module.MemoryManager, "__init__", init_storage)
        manager = FP8StaticPerHeadQuantQwen3NextMemManager.__new__(FP8StaticPerHeadQuantQwen3NextMemManager)
        fp8.FP8StaticPerHeadQuantMemManager.__init__(
            manager, 8, torch.bfloat16, max(4 // tp_size, 1), 256, 2 + draft_layers
        )
        return manager

    return load


@pytest.mark.parametrize("tp_size,rank", [(tp, rank) for tp in [1, 2, 4] for rank in range(tp)])
def test_native_per_head_calibration_file_loading(load_calibration, tp_size, rank):
    target = load_calibration(tp_size=tp_size, rank=rank)
    view = target.get_kv_layout_view(8, 128, layer_start=2, layer_num=1)
    for manager, layout, rows in [
        (target, calibration_layout(2, 4, 256, 1), slice(0, 2)),
        (view, calibration_layout(1, 8, 128, 101), slice(2, 3)),
    ]:
        heads, local = layout["num_head"], manager.head_num
        start, end = rank * local, (rank + 1) * local
        kv = torch.tensor(layout["scales"])
        q = torch.tensor(layout["q_calibration"]["scales"])
        torch.testing.assert_close(
            manager.scales[rows], torch.cat((kv[:, start:end], kv[:, heads + start : heads + end]), -1)
        )
        torch.testing.assert_close(manager.q_scales[rows], q[:, start:end])
    assert view.kv_buffer.data_ptr() == target.kv_buffer.data_ptr()
    with pytest.raises(ValueError, match="Missing FP8 calibration"):
        target.get_kv_layout_view(8, 128, layer_start=1, layer_num=2)


@pytest.mark.parametrize("tp_size,rank", [(1, 0), (4, 3), (8, 7)])
def test_layout_file_can_load_target_only_and_replicated_heads(load_calibration, tp_size, rank):
    manager = load_calibration(tp_size=tp_size, rank=rank, draft_layers=0)
    start = rank // max(tp_size // 4, 1) * manager.head_num
    q = torch.tensor(calibration_layout(2, 4, 256, 1)["q_calibration"]["scales"])
    torch.testing.assert_close(manager.q_scales, q[:, start : start + manager.head_num])


def test_legacy_calibration_does_not_invent_native_head_scales(load_calibration):
    cfg = calibration_layout(3, 4, 256, 1)
    target = load_calibration(cfg)
    torch.testing.assert_close(target.scales, torch.tensor(cfg["scales"]))
    assert target.get_kv_layout_view(4, 256) is target
    with pytest.raises(ValueError, match="No per-head FP8 calibration"):
        target.get_kv_layout_view(8, 128, layer_start=2, layer_num=1)


@pytest.mark.parametrize(
    "fault", ["kv_shape", "q_shape", "missing_q", "nonfinite", "negative", "coverage", "target_layout"]
)
def test_layout_calibration_file_rejects_invalid_metadata(load_calibration, fault):
    cfg = {"num_layers": 3, "layouts": [calibration_layout(2, 4, 256, 1), calibration_layout(1, 8, 128, 101)]}
    layout = cfg["layouts"][1]
    if fault == "kv_shape":
        layout["scales"][0].pop()
    elif fault == "q_shape":
        layout["q_calibration"]["scales_shape"] = [1, 4]
    elif fault == "missing_q":
        del layout["q_calibration"]
    elif fault == "nonfinite":
        layout["scales"][0][0] = float("nan")
    elif fault == "negative":
        layout["q_calibration"]["scales"][0][0] = -1
    elif fault == "coverage":
        cfg["layouts"].pop()
    else:
        cfg["layouts"][0] = calibration_layout(2, 8, 128, 1)
    with pytest.raises(ValueError):
        load_calibration(cfg)


@pytest.mark.parametrize("model_name", ["dspark", "dflash"])
def test_model_selects_its_calibrated_layer_range(load_calibration, model_name):
    from lightllm.models.qwen3_5_dspark.model import Qwen3_5DSparkModel
    from lightllm.models.qwen3_5_dflash.model import Qwen3_5DFlashModel

    cls = Qwen3_5DSparkModel if model_name == "dspark" else Qwen3_5DFlashModel
    model = cls.__new__(cls)
    model.main_model = SimpleNamespace(mem_manager=load_calibration())
    model.mtp_previous_draft_models = []
    model.config = {"num_key_value_heads": 8, "head_dim": 128, "n_layer": 1}
    model._init_mem_manager()
    torch.testing.assert_close(
        model.mem_manager.scales[2], torch.tensor(calibration_layout(1, 8, 128, 101)["scales"])[0]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_loaded_native_scales_quantize_and_survive_pd_transfer(load_calibration):
    source = load_calibration(device="cuda")
    draft = source.get_kv_layout_view(8, 128, layer_start=2, layer_num=1)
    kv = torch.randn((2, 16, 128), dtype=torch.bfloat16, device="cuda")
    slots = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    draft.operator.copy_kv_to_mem_manager(8, slots, kv)
    page = torch.zeros((1, 2, 3, 8, 256), dtype=torch.uint8, device="cuda")
    source.kv_move_buffer = page
    source._buffer_mem_indexes_tensors = [torch.empty(2, dtype=torch.int64, pin_memory=True)]
    source.write_mem_to_page_kv_move_buffer([1, 2], 0, 0, [source], 1)
    destinations = [load_calibration(tp_size=4, rank=rank, device="cuda") for rank in range(4)]
    destinations[0].kv_move_buffer = page
    destinations[0]._buffer_mem_indexes_tensors = [torch.empty(2, dtype=torch.int64, pin_memory=True)]
    destinations[0].read_page_kv_move_buffer_to_mem([3, 4], 0, 0, destinations, 4)
    scales = draft.scales[2].view(1, 16, 1)
    expected = (kv.float() / scales).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scales
    for rank, manager in enumerate(destinations):
        view = manager.get_kv_layout_view(8, 128, layer_start=2, layer_num=1)
        k, v = view.get_att_input_params(8)
        actual = torch.cat((k[3:5], v[3:5]), 1).view(torch.float8_e4m3fn).float() * view.scales[2].view(1, 4, 1)
        torch.testing.assert_close(
            actual, torch.cat((expected[:, rank * 2 : rank * 2 + 2], expected[:, 8 + rank * 2 : 10 + rank * 2]), 1)
        )


def test_later_draft_uses_its_own_calibration_rows(load_calibration):
    from lightllm.models.qwen3_5_dspark.model import Qwen3_5DSparkModel

    cfg = {
        "num_layers": 5,
        "num_target_layers": 2,
        "num_draft_layers": 3,
        "layouts": [
            calibration_layout(2, 4, 256, 1),
            calibration_layout(1, 8, 128, 101),
            calibration_layout(2, 8, 128, 201),
        ],
    }
    model = Qwen3_5DSparkModel.__new__(Qwen3_5DSparkModel)
    model.main_model = SimpleNamespace(mem_manager=load_calibration(cfg, draft_layers=3))
    model.mtp_previous_draft_models = [SimpleNamespace(layers_infer=[object()])]
    model.config = {"num_key_value_heads": 8, "head_dim": 128, "n_layer": 2}
    model._init_mem_manager()
    torch.testing.assert_close(model.mem_manager.scales[3:5], torch.tensor(cfg["layouts"][2]["scales"]))
    torch.testing.assert_close(
        model.mem_manager.q_scales[3:5], torch.tensor(cfg["layouts"][2]["q_calibration"]["scales"])
    )
