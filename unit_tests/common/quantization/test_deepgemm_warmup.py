from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel import TpPartBaseModel
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight
from lightllm.common.quantization import deepgemm
from lightllm.common.quantization.quantize_method import WeightPack
from lightllm.utils import device_utils


@pytest.mark.parametrize("max_m", [1, 4, 17, 33, 65, 513, 8192])
def test_dense_warmup_covers_partial_tiles_without_changing_weights(monkeypatch, max_m):
    for name in ("zeros", "ones", "empty"):
        original = getattr(torch, name)

        def cpu_tensor(*args, _original=original, **kwargs):
            kwargs["device"] = "cpu"
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, cpu_tensor)
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    weight = WeightPack(torch.ones((128, 256), dtype=torch.float8_e4m3fn), torch.ones((1, 2)))
    before = weight.weight.float().clone()

    def gemm(lhs, rhs, output):
        quant, scale = lhs
        m = quant.shape[0]
        assert quant.shape == (m, 256) and output.shape == (m, 128)
        assert output.dtype == torch.bfloat16
        assert scale.shape == (m, 2) and scale.stride() == (1, (m + 3) // 4 * 4)
        assert torch.all(quant.float() == 0) and torch.all(scale == 1)
        assert rhs[0] is weight.weight and rhs[1] is weight.weight_scale
        events.append(m)

    monkeypatch.setattr(deepgemm, "_deepgemm_fp8_nt", gemm)
    method = deepgemm.DeepGEMMFP8w8a8B128QuantizationMethod.__new__(deepgemm.DeepGEMMFP8w8a8B128QuantizationMethod)
    method.block_size = 128
    method.warmup(weight, max_m, torch.bfloat16)
    covered = set(events[:-1])
    assert events[-1] == "sync" and min(covered) >= 1 and max(covered) == max_m
    for m in range(1, max_m + 1):
        representative = 16 if m <= 16 else 32 if m <= 32 else (m + 63) // 64 * 64
        assert min(representative, max_m) in covered
    assert torch.equal(weight.weight.float(), before)


@pytest.mark.parametrize("mode,sm90", [("prefill", True), ("decode", True), ("prefill", False)])
def test_model_warms_selected_dense_shapes_once_and_propagates_failures(monkeypatch, mode, sm90):
    monkeypatch.setattr(device_utils, "is_sm90_gpu", lambda: sm90)
    calls = []
    method = deepgemm.DeepGEMMFP8w8a8B128QuantizationMethod.__new__(deepgemm.DeepGEMMFP8w8a8B128QuantizationMethod)

    def dense(n, k):
        weight = ROWMMWeight.__new__(ROWMMWeight)
        weight.quant_method = method
        weight.mm_param = WeightPack(torch.empty((n, k)))
        weight.data_type_ = torch.bfloat16
        return weight

    first, duplicate, other = dense(128, 256), dense(128, 256), dense(256, 128)
    unquantized = dense(64, 128)
    unquantized.quant_method = object()
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.args = SimpleNamespace(run_mode=mode)
    model.batch_max_tokens = 513
    model.pre_post_weight = SimpleNamespace(named_weights=lambda: {"proj": first})
    model.trans_layers_weight = [
        SimpleNamespace(
            named_weights=lambda: {"duplicate": duplicate, "other": other, "bf16": unquantized, "experts": object()}
        )
    ]
    monkeypatch.setattr(method, "warmup", lambda pack, rows, dtype: calls.append((pack, rows, dtype)))
    model._kernel_warmup()
    if mode != "prefill" or not sm90:
        assert calls == []
        return
    assert [pack for pack, _, _ in calls] == [first.mm_param, other.mm_param]
    assert all(rows == 513 and dtype == torch.bfloat16 for _, rows, dtype in calls)

    def fail(*args):
        raise RuntimeError("native compilation failed")

    monkeypatch.setattr(method, "warmup", fail)
    with pytest.raises(RuntimeError, match="native compilation failed"):
        model._kernel_warmup()


def test_dspark_model_owned_projection_is_warmed_and_keeps_resident_filter(monkeypatch):
    from lightllm.models.deepseek_v4_dspark.model import DeepseekV4DSparkModel

    monkeypatch.setattr(device_utils, "is_sm90_gpu", lambda: True)
    method = deepgemm.DeepGEMMFP8w8a8B128QuantizationMethod.__new__(deepgemm.DeepGEMMFP8w8a8B128QuantizationMethod)
    projection = ROWMMWeight.__new__(ROWMMWeight)
    projection.quant_method = method
    projection.mm_param = WeightPack(torch.empty((1536, 5120)))
    projection.data_type_ = torch.bfloat16
    host = SimpleNamespace(supports_resident_cache=False)
    model = DeepseekV4DSparkModel.__new__(DeepseekV4DSparkModel)
    model.args = SimpleNamespace(run_mode="prefill")
    model.batch_max_tokens = 8192
    model.pre_post_weight = SimpleNamespace(named_weights=lambda: {"host": host})
    model.trans_layers_weight = []
    model.context_wkv_weight = projection
    calls = []
    monkeypatch.setattr(method, "warmup", lambda pack, rows, dtype: calls.append((pack, rows, dtype)))

    TpPartBaseModel._kernel_warmup(model)

    assert calls == [(projection.mm_param, 8192, torch.bfloat16)]
    assert model.named_weights() == {"pre_post.host": host, "context_wkv": projection}
    assert model.named_resident_weights() == {"context_wkv": projection}
