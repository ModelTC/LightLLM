import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def test_speculative_decode_groups_q_scales_by_request():
    source = Path("lightllm/common/basemodel/attention/fa3/fp8.py")
    tree = ast.parse(source.read_text(), filename=str(source))
    method = next(
        item
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Fp8Fa3DecodeAttState"
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == "_fp8_decode_att"
    )
    seen = {}

    def q_per_head(q, seq_lens, cu_seqlens, token_batch_ids=None):
        seen["q_per_head"] = (tuple(q.shape), tuple(seq_lens.tolist()), tuple(token_batch_ids.tolist()))
        return q, torch.ones((seq_lens.numel(), q.shape[1]), dtype=torch.float32)

    def flash(**kwargs):
        seen["q_descale"] = tuple(kwargs["q_descale"].shape)
        return kwargs["q"]

    namespace = {"torch": torch, "q_per_head_fp8_quant": q_per_head, "flash_attn_with_kvcache": flash, "scaled_fp8_quant": None}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    state = SimpleNamespace(
        backend=SimpleNamespace(_find_layer_index=lambda **_: 0),
        page_table=torch.zeros((2, 1), dtype=torch.int32),
        b_att_seq_len=torch.tensor([3, 3], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 3, 6], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 3, 6], dtype=torch.int32),
        decode_max_q_seq_len=3,
        causal=True,
        k_descale=[torch.ones((2, 1))],
        v_descale=[torch.ones((2, 1))],
    )
    namespace["_fp8_decode_att"](
        state,
        torch.zeros((6, 1, 128)),
        torch.zeros((6, 1, 128), dtype=torch.uint8),
        torch.zeros((6, 1, 128), dtype=torch.uint8),
    )
    assert seen["q_per_head"] == ((6, 1, 128), (3, 3), (0, 0, 0, 1, 1, 1))
    assert seen["q_descale"] == (2, 1)
