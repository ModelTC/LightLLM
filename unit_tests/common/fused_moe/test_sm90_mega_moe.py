from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.fused_moe import sm90_fp8_mega_moe as mega
from lightllm.common.quantization.quantize_method import WeightPack


@pytest.fixture
def module(monkeypatch):
    module = SimpleNamespace(
        fp8_mega_moe=lambda *args, **kwargs: None,
        mega_moe_pre_dispatch_sm90=lambda *args, **kwargs: None,
        get_symm_buffer_for_sm90_mega_moe=lambda *args: None,
    )
    monkeypatch.setattr(mega, "get_sm90_mega_moe_module", lambda: module)
    return module


def packs():
    # Distinct row bytes expose a wrong gate/up permutation; scales must stay
    # in the original checkpoint order even though the weight rows move.
    raw = torch.arange(256).to(torch.uint8).view(1, 256, 1).expand(2, 256, 128).contiguous()
    w13 = WeightPack(raw.view(torch.float8_e4m3fn), torch.arange(4).float().view(2, 2, 1))
    w2 = WeightPack(torch.zeros(2, 128, 128).to(torch.float8_e4m3fn), torch.ones(2, 1, 1))
    return w13, w2


def test_weight_preparation_retains_storage_and_checkpoint_scales(module):
    w13, w2 = packs()
    pointers = [t.data_ptr() for t in (w13.weight, w13.weight_scale, w2.weight, w2.weight_scale)]
    original_scales = w13.weight_scale.clone()
    expected = [row for start in range(0, 128, 8) for base in (start, start + 128) for row in range(base, base + 8)]
    prepared = mega.prepare_sm90_mega_moe_weights(w13, w2)
    assert w13.weight.view(torch.uint8)[0, :, 0].tolist() == expected
    torch.testing.assert_close(w13.weight_scale, original_scales)
    assert [t.data_ptr() for pair in prepared for t in pair] == pointers
    assert mega.prepare_sm90_mega_moe_weights(w13, w2) is prepared
    assert w13.weight.view(torch.uint8)[0, :, 0].tolist() == expected


@pytest.mark.parametrize("bad", ["dtype", "scale", "shape"])
def test_reject_incompatible_weights_before_mutation(module, bad):
    w13, w2 = packs()
    original = w13.weight.view(torch.uint8).clone()
    if bad == "dtype":
        w2.weight = w2.weight.float()
    elif bad == "scale":
        w2.weight_scale = w2.weight_scale.to(torch.int32)
    else:
        w2.weight = torch.zeros(2, 128, 256).to(torch.float8_e4m3fn)
        w2.weight_scale = torch.ones(2, 1, 2)
    with pytest.raises(ValueError):
        mega.prepare_sm90_mega_moe_weights(w13, w2)
    assert torch.equal(w13.weight.view(torch.uint8), original)


@pytest.mark.parametrize("tokens", [7, 1, 0])
def test_forward_passes_full_padding_buffers_and_participates_when_empty(module, tokens):
    w13, w2 = packs()
    x = torch.empty(tokens, 128, dtype=torch.bfloat16)
    ids = torch.zeros(tokens, 1, dtype=torch.int64)
    weights = torch.full((tokens, 1), 2.5)
    buf = SimpleNamespace(x=object(), x_sf=object(), topk_idx=object(), topk_weights=object())
    stats = torch.zeros(2, dtype=torch.int32)
    calls = []

    def dispatch(*args, **kwargs):
        assert args[0] is x and args[2] is weights
        assert args[1].dtype == torch.int32 and torch.equal(args[1], ids)
        assert args[3:] == (buf.x, buf.x_sf, buf.topk_idx, buf.topk_weights)
        assert kwargs == dict(num_tokens=tokens, group_size=128, routed_scaling_factor=1.0)
        calls.append("dispatch")

    def kernel(out, l1, l2, buffer, **kwargs):
        assert buffer is buf and out.shape == (max(tokens, 1), 128)
        assert kwargs["cumulative_local_expert_recv_stats"] is stats
        assert kwargs["activation_clamp"] == 10.0
        assert kwargs["recipe"] == (128, 128, 128)
        out.fill_(3)
        calls.append("kernel")

    module.mega_moe_pre_dispatch_sm90 = dispatch
    module.fp8_mega_moe = kernel
    actual = mega.sm90_mega_moe_forward(x, w13, w2, weights, ids, buf, stats, 10.0, torch.empty)
    assert calls == ["dispatch", "kernel"]
    assert actual.shape == x.shape and torch.all(actual == 3)


def test_dependency_failure_is_actionable(monkeypatch):
    # Reach the undecorated loader so no real CUDA dependency is imported.
    monkeypatch.setattr(mega.importlib, "import_module", lambda name: SimpleNamespace())
    with pytest.raises(RuntimeError, match="README_megamoe.md"):
        mega.get_sm90_mega_moe_module.__wrapped__()


def test_oversized_batch_is_rejected_before_dispatch(monkeypatch):
    from lightllm.common.basemodel.triton_kernel.fused_moe import grouped_fused_moe_ep as ep

    monkeypatch.setattr(ep.dist_group_manager, "ep_mega_moe_buffer", SimpleNamespace(num_max_tokens_per_rank=1))
    with pytest.raises(RuntimeError, match="exceeding"):
        ep.mega_moe_impl(torch.empty(2, 128), None, None, None, None, None, "fp8xfp8")


def test_weight_finalization_disables_legacy_overlap(monkeypatch):
    from lightllm.common.basemodel.triton_kernel.fused_moe import grouped_fused_moe_ep as ep
    from lightllm.common.quantization.deepgemm import DeepGEMMBaseQuantizationMethod

    quant = SimpleNamespace(mega_moe_mma_type=None)
    owner = SimpleNamespace(
        w13=None, w2=None, fuse_moe_impl=SimpleNamespace(supports_decode_shared_expert_overlap=True)
    )

    def prepare(_w13, _w2, method):
        method.mega_moe_mma_type = "fp8xfp8"

    monkeypatch.setattr(ep, "prepare_ep_moe_weights", prepare)
    DeepGEMMBaseQuantizationMethod.finalize_moe_weight(quant, owner)
    assert owner.fuse_moe_impl.supports_decode_shared_expert_overlap is False


@pytest.mark.parametrize("mode,capacity", [("normal", 64), ("decode", 64), ("prefill", 8)])
def test_buffer_initialization_covers_decode_graph_capacity(monkeypatch, mode, capacity):
    from unittest.mock import Mock

    from lightllm.distributed import communication_op as comm
    from lightllm.server.core.objs.start_args_type import StartArgs

    args = StartArgs(enable_ep_moe=True, ep_moe_backend="megamoe", run_mode=mode)
    monkeypatch.setattr(comm, "get_env_start_args", lambda: args)
    monkeypatch.setattr(comm, "get_global_world_size", lambda: 2)
    monkeypatch.setattr(comm, "get_deepep_num_max_dispatch_tokens_per_rank_prefill", lambda: 8)
    monkeypatch.setattr(comm, "get_deepep_num_max_dispatch_tokens_per_rank_decode", lambda: 64)
    monkeypatch.setattr(comm, "is_sm90_gpu", lambda: True)
    monkeypatch.setattr(comm, "is_sm100_gpu", lambda: False)
    monkeypatch.setattr(comm, "HAS_DEEPEP", True)
    group = object()
    monkeypatch.setattr(comm.dist, "new_group", lambda *args: group)
    monkeypatch.setattr(comm.dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(comm.torch.cuda, "current_device", lambda: 0)
    elastic = SimpleNamespace(get_theoretical_num_sms=lambda *args: 16)
    monkeypatch.setattr(comm, "deep_ep", SimpleNamespace(ElasticBuffer=lambda *args, **kwargs: elastic))
    allocate = Mock(return_value=object())
    module = SimpleNamespace(get_symm_buffer_for_sm90_mega_moe=allocate)
    monkeypatch.setattr(mega, "get_sm90_mega_moe_module", lambda: module)
    manager = comm.DistributeGroupManager()
    manager._set_num_sms_for_deep_gemm = Mock()
    manager.new_deepep_group(288, 4096, {comm.FP8_MOE_QUANT_METHOD}, 8, 2048)
    allocate.assert_called_once_with(group, 288, capacity, 8, 4096, 2048)
    assert manager.ep_low_latency_buffer is None
    assert manager.ep_prefill_moe_workspace is None
    assert manager.ep_mega_moe_mma_type == "fp8xfp8"
