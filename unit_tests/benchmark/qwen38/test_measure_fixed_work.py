from copy import deepcopy

import pytest

from test.benchmark.qwen38.measure_fixed_work import summarize_fixed_work
from test.benchmark.service.stream_cohort import StreamingWorkload, run_cohort
from unit_tests.benchmark.service.test_stream_cohort import observation


def fixture():
    work = StreamingWorkload("prompt", [12, 13], [20, 21])
    result = observation(1, [2, 3], 4, ids=[22, 23])
    result["metrics"]["cached_input_tokens"] = 0
    return work, result


def test_id_mismatch_keeps_exact_gate_failed_but_measures_equal_work(tmp_path):
    work, result = fixture()
    cohort = run_cohort([work], 1, lambda *_: result, tmp_path)
    before = deepcopy(cohort)
    metrics = summarize_fixed_work(cohort, [work])
    assert metrics["output_tokens_per_s"] == pytest.approx(2 / 3)
    assert cohort == before
    assert cohort["status"] == "failed" and cohort["metrics"] is None


@pytest.mark.parametrize("cached", [None, True, -1, 1])
def test_unknown_or_nonzero_cache_cannot_produce_timing(tmp_path, cached):
    work, result = fixture()
    result["metrics"]["cached_input_tokens"] = cached
    cohort = run_cohort([work], 1, lambda *_: result, tmp_path)
    with pytest.raises(ValueError):
        summarize_fixed_work(cohort, [work])


@pytest.mark.parametrize("fault", ["transport", "input_length", "output_length", "timeline", "missing"])
def test_invalid_work_never_produces_timing(tmp_path, fault):
    work, result = fixture()
    if fault == "input_length":
        result["metrics"]["input_tokens"] = 3
    elif fault == "output_length":
        result["token_ids"] = [22]
    elif fault == "timeline":
        result["timeline"]["ended"] = 0

    def request(*_):
        if fault == "transport":
            raise TimeoutError("timed out")
        return result

    cohort = run_cohort([work], 1, request, tmp_path)
    if fault == "missing":
        cohort["records"] = []
    with pytest.raises(ValueError):
        summarize_fixed_work(cohort, [work])
