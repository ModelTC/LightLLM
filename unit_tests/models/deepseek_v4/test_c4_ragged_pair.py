import ast
from pathlib import Path

from lightllm.models.deepseek_v4.c4_ragged_pair import build_c4_ragged_pair_plan


def test_even_requests_use_lightweight_pair_sentinel():
    plan = build_c4_ragged_pair_plan([128, 128], [0, 0], 256)
    assert plan is not None
    assert plan.kind == "all_even"
    assert plan.packed_to_original == ()
    assert plan.padded_rows == 0


def test_ragged_pairs_stay_within_request_and_restore_original_rows():
    plan = build_c4_ragged_pair_plan([127, 129], [0, 0], 256)
    assert plan is not None
    assert plan.kind == "ragged"
    assert plan.padded_rows == 2
    assert len(plan.packed_to_original) == 258
    assert [plan.packed_to_original[index] for index in plan.original_to_packed] == list(range(256))
    assert all(
        plan.packed_request[index] == plan.packed_request[index + 1]
        for index in range(0, len(plan.packed_request), 2)
    )


def test_invalid_or_high_padding_requests_fall_back():
    assert build_c4_ragged_pair_plan([1] * 256, [0] * 256, 256) is None
    assert build_c4_ragged_pair_plan([128, 128], [0, 0], 255) is None
    assert build_c4_ragged_pair_plan([256.0], [0], 256) is None
    assert build_c4_ragged_pair_plan([256], [257], -1) is None


def test_c4_pair_plan_is_attached_only_by_prefill_role():
    model_source = Path(__file__).parents[3] / "lightllm/models/deepseek_v4/model.py"
    tree = ast.parse(model_source.read_text())
    create_state = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_create_inferstate")
    model_class = ast.ClassDef(
        name="LoadedModel",
        bases=[ast.Name(id="MinimalSuper", ctx=ast.Load())],
        keywords=[],
        body=[create_state],
        decorator_list=[],
    )
    module = ast.Module(body=[model_class], type_ignores=[])
    ast.fix_missing_locations(module)

    class MinimalSuper:
        def _create_inferstate(self, model_input, microbatch_index=0):
            return type("State", (), {})()

    namespace = {"MinimalSuper": MinimalSuper, "ModelInput": object}
    exec(compile(module, str(model_source), "exec"), namespace)
    LoadedModel = namespace["LoadedModel"]
    calls = []
    plan = type("Plan", (), {"kind": "all_even"})()

    def spy_plan(model_input):
        calls.append(model_input)
        return plan

    LoadedModel._c4_pair_prefill_plan = staticmethod(spy_plan)
    prefill_model = LoadedModel()
    prefill_model.run_mode = "prefill"
    prefill_input = object()
    prefill_state = prefill_model._create_inferstate(prefill_input)
    assert calls == [prefill_input]
    assert prefill_state.dsv4_c4_ragged_pair_plan is plan
    assert prefill_state.dsv4_c4_pair_eligible

    decode_model = LoadedModel()
    decode_model.run_mode = "decode"
    decode_state = decode_model._create_inferstate(object())
    assert calls == [prefill_input]
    assert decode_state.dsv4_c4_ragged_pair_plan is None
    assert not decode_state.dsv4_c4_pair_eligible
