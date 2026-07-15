import torch

from lightllm.server.router.model_infer.mode_backend.chunked_prefill.mtp_fused_decode_graph import (
    MTPFusedDecodeGraph,
)


def _make_graph(buf_size: int = 16) -> MTPFusedDecodeGraph:
    g = object.__new__(MTPFusedDecodeGraph)
    g.b_position_delta = torch.zeros(buf_size, dtype=torch.int64)
    g.b_position_delta_pin = torch.zeros(buf_size, dtype=torch.int64)
    g._position_delta_rows = 0
    return g


def test_stale_delta_cleared_on_shrink_then_grow():
    g = _make_graph()

    # step A: batch 8, 第 5 行有 image delta
    g.b_position_delta_pin[:8] = torch.tensor([0, 0, 0, 0, 0, 100, 0, 0])
    g._flush_position_delta(has_delta=True, batch_size=8)
    assert g.b_position_delta[5].item() == 100

    # step B: batch 4, 纯文本 (staging 只刷新前 4 行 pin)
    g.b_position_delta_pin[:4].zero_()
    g._flush_position_delta(has_delta=False, batch_size=4)

    # step C: batch 8, 纯文本 -> 第 5 行不能读到 step A 的残留 delta
    g.b_position_delta_pin[:8].zero_()
    g._flush_position_delta(has_delta=False, batch_size=8)
    assert g.b_position_delta[:8].eq(0).all(), g.b_position_delta[:8]


def test_delta_batches_keep_correct_values():
    g = _make_graph()

    g.b_position_delta_pin[:8] = torch.tensor([0, 0, 0, 0, 0, 100, 0, 0])
    g._flush_position_delta(has_delta=True, batch_size=8)

    # 更小的带 delta batch: 前缀是新值, 尾部旧值必须被冲掉
    g.b_position_delta_pin[:4] = torch.tensor([0, 0, 7, 0])
    g._flush_position_delta(has_delta=True, batch_size=4)
    assert g.b_position_delta[2].item() == 7
    assert g.b_position_delta[4:8].eq(0).all()

    # 干净后不再需要上传, 缓冲保持全零
    g.b_position_delta_pin[:4].zero_()
    g._flush_position_delta(has_delta=False, batch_size=4)
    g._flush_position_delta(has_delta=False, batch_size=8)
    assert g.b_position_delta.eq(0).all()
