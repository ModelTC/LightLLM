from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.attention.nsa import flashmla_sparse


def _disabled_modes():
    return SimpleNamespace(
        enable_tpsp_mix_mode=False,
        enable_prefill_cudagraph=False,
        enable_prefill_microbatch_overlap=False,
        enable_prefill_decode_mixed=False,
    )


def _infer_state(world_size=8):
    return SimpleNamespace(
        dist_group=SimpleNamespace(dp_world_size=world_size),
        max_cache_len=0,
        need_dp_prefill_balance=False,
        use_replicated_attention_ep=False,
    )


def test_copy_received_head_shards_makes_token_major_global_heads():
    # Two source ranks sent the same destination's two-token chunk.
    received = torch.tensor([[[10]], [[11]], [[20]], [[21]]])
    output = torch.empty((2, 2, 1), dtype=received.dtype)

    flashmla_sparse._copy_received_head_shards(received, output, world_size=2)

    torch.testing.assert_close(output, torch.tensor([[[10], [20]], [[11], [21]]]))


def test_copy_token_shard_for_head_scatter_makes_destination_blocks():
    output = torch.tensor([[[10], [20]], [[11], [21]]])
    send = torch.empty((4, 1, 1), dtype=output.dtype)

    flashmla_sparse._copy_token_shard_for_head_scatter(output, send, world_size=2)

    torch.testing.assert_close(send, torch.tensor([[[10]], [[11]], [[20]], [[21]]]))


def test_tp_head_token_transpose_accepts_validated_glm_tp8_layout(monkeypatch):
    monkeypatch.setattr(flashmla_sparse, "get_env_start_args", _disabled_modes)
    q = torch.empty((4096, 8, 2))

    assert flashmla_sparse._should_use_tp_head_token_transpose(q, _infer_state(), required_heads=64)


@pytest.mark.parametrize(
    ("tokens", "heads", "world_size", "state_change", "mode_change"),
    [
        (2048, 8, 8, {}, {}),
        (4097, 8, 8, {}, {}),
        (4096, 16, 8, {}, {}),
        (4096, 8, 1, {}, {}),
        (4096, 8, 8, {"max_cache_len": 1}, {}),
        (4096, 8, 8, {"need_dp_prefill_balance": True}, {}),
        (4096, 8, 8, {"use_replicated_attention_ep": True}, {}),
        (4096, 8, 8, {}, {"enable_tpsp_mix_mode": True}),
        (4096, 8, 8, {}, {"enable_prefill_cudagraph": True}),
        (4096, 8, 8, {}, {"enable_prefill_microbatch_overlap": True}),
        (4096, 8, 8, {}, {"enable_prefill_decode_mixed": True}),
    ],
)
def test_tp_head_token_transpose_rejects_unvalidated_layouts(
    monkeypatch,
    tokens,
    heads,
    world_size,
    state_change,
    mode_change,
):
    modes = _disabled_modes()
    for name, value in mode_change.items():
        setattr(modes, name, value)
    monkeypatch.setattr(flashmla_sparse, "get_env_start_args", lambda: modes)
    state = _infer_state(world_size)
    for name, value in state_change.items():
        setattr(state, name, value)
    q = torch.empty((tokens, heads, 2))

    assert not flashmla_sparse._should_use_tp_head_token_transpose(q, state, required_heads=64)
