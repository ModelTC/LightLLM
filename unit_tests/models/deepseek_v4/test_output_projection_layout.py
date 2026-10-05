import ast
from pathlib import Path
from types import SimpleNamespace

import torch


SOURCE = Path(__file__).parents[3] / "lightllm/models/deepseek_v4/layer_infer/transformer_layer_infer.py"


def _load_get_o():
    tree = ast.parse(SOURCE.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_get_o")
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "rotary_emb_fwd": lambda *args, **kwargs: None,
        "DeepseekV4InferStateInfo": object,
        "DeepseekV4TransformerLayerWeight": object,
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["_get_o"]


class _GroupedProjection:
    def __init__(self):
        self.weight = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, -1.0], [0.5, 3.0]],
            ]
        )
        self.used_out = []

    def bmm(self, value, out=None):
        self.used_out.append(out is not None)
        if out is not None:
            return torch.bmm(value, self.weight, out=out)
        return torch.bmm(value, self.weight)


class _IdentityProjection:
    def mm(self, value):
        return value


class _Layer:
    tp_groups = 2
    qk_rope_head_dim = 1

    def __init__(self, decode_role):
        self._dsv4_decode_role = decode_role

    def _select_rope(self, infer_state):
        return None, None

    def alloc_tensor(self, shape, dtype, device):
        return torch.empty(shape, dtype=dtype, device=device)

    def _tpsp_reduce(self, input, infer_state):
        return input


def _run_case(get_o, tokens, prefill, decode_role):
    layer = _Layer(decode_role)
    grouped = _GroupedProjection()
    weights = SimpleNamespace(o_proj_fp8=False, wo_a_=grouped, wo_b_=_IdentityProjection())
    state = SimpleNamespace(is_prefill=prefill)
    input_value = torch.arange(tokens * 4, dtype=torch.float32).reshape(tokens, 2, 2)
    result = get_o(layer, input_value.clone(), state, weights)
    expected = torch.einsum("tgi,gio->tgo", input_value, grouped.weight).reshape(tokens, -1)
    assert torch.equal(result, expected)
    return grouped.used_out


def test_output_projection_layout_uses_out_only_for_prefill_or_decode_role():
    get_o = _load_get_o()
    for tokens in (1, 3):
        assert _run_case(get_o, tokens, prefill=True, decode_role=False) == [True]
        assert _run_case(get_o, tokens, prefill=False, decode_role=False) == [False]
        assert _run_case(get_o, tokens, prefill=False, decode_role=True) == [True]
