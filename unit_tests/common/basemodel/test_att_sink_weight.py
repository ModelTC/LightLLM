import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights import TpAttSinkWeight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "heads,tp_size,rank,padding_multiple,data_type,start,end,padded_heads",
    [
        (128, 8, 0, 64, torch.float32, 0, 16, 64),
        (128, 8, 7, 64, torch.float32, 112, 128, 64),
        (128, 2, 1, 64, torch.float32, 64, 128, 64),
        (256, 2, 1, 64, torch.float32, 128, 256, 128),
        (12, 4, 2, None, torch.bfloat16, 6, 9, 3),
        (4, 8, 3, None, torch.bfloat16, 1, 2, 1),
    ],
)
def test_load_tp_sink_weight(monkeypatch, heads, tp_size, rank, padding_multiple, data_type, start, end, padded_heads):
    monkeypatch.setenv("LIGHTLLM_DP_WORLD_SIZE", str(tp_size))
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_DP", str(rank))
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    kwargs = {} if padding_multiple is None else {"head_padding_multiple": padding_multiple}
    sink = TpAttSinkWeight(all_q_head_num=heads, weight_name="attn_sink", data_type=data_type, **kwargs)
    assert sink.weight.shape == (padded_heads,)
    assert not sink.verify_load()
    storage_ptr = sink.weight.data_ptr()
    checkpoint = torch.arange(heads, dtype=torch.float64) + 0.5

    sink.load_hf_weights({"attn_sink": checkpoint})

    assert sink.verify_load()
    assert sink.weight.data_ptr() == storage_ptr
    assert sink.weight.dtype == data_type
    assert sink.weight.is_cuda
    torch.testing.assert_close(sink.weight[: end - start], checkpoint[start:end].to(device="cuda", dtype=data_type))
    assert torch.count_nonzero(sink.weight[end - start :]) == 0
