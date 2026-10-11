import ast
from pathlib import Path

from lightllm.models.deepseek_v4.c4_ragged_pair import build_c4_ragged_pair_plan


def _owners_from_lengths(seq, ready):
    owners = []
    for request, (s, r) in enumerate(zip(seq, ready)):
        owners.extend([request] * (s - r))
    return owners


def _assert_request_local(plan, owners):
    assert [plan.packed_to_original[index] for index in plan.original_to_packed] == list(range(len(owners)))
    for left, right in zip(plan.packed_to_original[::2], plan.packed_to_original[1::2]):
        assert owners[left] == owners[right]


def test_even_requests_use_lightweight_pair_sentinel():
    plan = build_c4_ragged_pair_plan([128, 128], [0, 0], 256)
    assert plan is not None
    assert plan.kind == "all_even"
    assert plan.packed_to_original == ()
    assert plan.original_to_packed == ()


def test_ragged_pairs_derive_request_boundaries_from_original_lengths():
    seq, ready = [129, 131], [0, 0]
    plan = build_c4_ragged_pair_plan(seq, ready, 260)
    assert plan is not None and plan.kind == "ragged"
    assert len(plan.packed_to_original) == 262
    _assert_request_local(plan, _owners_from_lengths(seq, ready))


def test_zero_length_requests_and_single_odd_request_keep_pairs_local():
    seq, ready = [0, 129, 0, 131], [0, 0, 0, 0]
    plan = build_c4_ragged_pair_plan(seq, ready, 260)
    assert plan is not None and plan.kind == "ragged"
    _assert_request_local(plan, _owners_from_lengths(seq, ready))
    singleton = build_c4_ragged_pair_plan([257], [0], 257)
    assert singleton is not None and singleton.kind == "ragged"
    _assert_request_local(singleton, _owners_from_lengths([257], [0]))


def test_high_padding_stays_request_local_and_invalid_inputs_raise():
    assert build_c4_ragged_pair_plan([1] * 256, [0] * 256, 256).kind == "ragged"
    for seq, ready, count in (([128, 128], [0, 0], 255), ([256.0], [0], 256), ([256], [257], -1)):
        try:
            build_c4_ragged_pair_plan(seq, ready, count)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid C4 pair input must raise")


def test_empty_batch_is_a_valid_empty_pair_plan():
    plan = build_c4_ragged_pair_plan([], [], 0)
    assert plan.kind == "all_even"


def test_c4_pair_plan_is_attached_only_by_prefill_role():
    model_source = Path(__file__).parents[3] / "lightllm/models/deepseek_v4/model.py"
    tree = ast.parse(model_source.read_text())
    create_state = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_create_inferstate"
    )
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
    decode_model = LoadedModel()
    decode_model.run_mode = "decode"
    decode_state = decode_model._create_inferstate(object())
    assert calls == [prefill_input]
    assert decode_state.dsv4_c4_ragged_pair_plan is None
