import json

import pytest

from test.benchmark.service.cache_capacity import plan_vllm_capacity


def metrics(**overrides):
    fields = dict(
        engine="0",
        num_gpu_blocks="12345",
        kv_cache_max_concurrency=str(12345 / 258),
        kv_cache_size_tokens=str(int(12345 / 258 * 4096)),
        block_size="16",
        mamba_block_size="256",
        mamba_cache_mode="align",
        mamba_ssm_cache_dtype="float32",
    )
    fields.update(overrides)
    labels = ",".join(f"{key}={json.dumps(value)}" for key, value in fields.items())
    return f"# TYPE vllm:cache_config_info gauge\nvllm:cache_config_info{{{labels}}} 1\n"


def test_hybrid_capacity_does_not_use_attention_block_size_alone():
    plan = plan_vllm_capacity(metrics(), 4096, 16384)
    assert plan["blocks_per_full_request"] == 258
    assert plan["num_gpu_blocks_override"] == 1032
    assert plan["cache_geometry"]["mamba_cache_mode"] == "align"


def test_unverified_pool_growth_is_not_planned():
    with pytest.raises(ValueError, match="only shrinks"):
        plan_vllm_capacity(metrics(), 4096, 196608)


@pytest.mark.parametrize("text", ["", metrics() + metrics(engine="1")])
def test_missing_or_ambiguous_engines_cannot_yield_capacity_plan(text):
    with pytest.raises(ValueError, match="exactly one"):
        plan_vllm_capacity(text, 4096, 16384)


@pytest.mark.parametrize(
    "fields",
    [
        dict(num_gpu_blocks="None"),
        dict(num_gpu_blocks="0"),
        dict(kv_cache_max_concurrency="nan"),
        dict(kv_cache_max_concurrency="inf"),
        dict(kv_cache_size_tokens="9"),
        dict(num_gpu_blocks="12346"),
        dict(kv_cache_max_concurrency="0"),
    ],
)
def test_uninitialized_or_inconsistent_runtime_metrics_are_rejected(fields):
    with pytest.raises(ValueError):
        plan_vllm_capacity(metrics(**fields), 4096, 16384)


@pytest.mark.parametrize("length,budget", [(True, 16384), (0, 16384), (4096, -1), (4096, 16385)])
def test_invalid_or_partial_context_budgets_are_rejected(length, budget):
    with pytest.raises(ValueError):
        plan_vllm_capacity(metrics(), length, budget)
