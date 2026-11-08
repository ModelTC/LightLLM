import math

from prometheus_client.parser import text_string_to_metric_families


def plan_vllm_capacity(metrics_text, model_length, target_tokens):
    if any(type(value) is not int or value < 1 for value in (model_length, target_tokens)):
        raise ValueError("Require positive integer context and token budgets")
    if target_tokens % model_length:
        raise ValueError("Token budget must contain whole full-context requests")
    samples = [
        sample
        for family in text_string_to_metric_families(metrics_text)
        for sample in family.samples
        if sample.name == "vllm:cache_config_info"
    ]
    if len(samples) != 1 or samples[0].value != 1:
        raise ValueError("Require exactly one initialized cache-info sample")
    fields = samples[0].labels
    try:
        blocks = int(fields["num_gpu_blocks"])
        concurrency = float(fields["kv_cache_max_concurrency"])
        reported_tokens = int(fields["kv_cache_size_tokens"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Initialized block count and group-aware capacity are required") from error
    if blocks < 1 or not math.isfinite(concurrency) or concurrency <= 0 or reported_tokens < 1:
        raise ValueError("Runtime cache geometry must be finite and positive")
    if reported_tokens != int(concurrency * model_length):
        raise ValueError("Reported capacity does not match the recorded context length")
    if target_tokens > reported_tokens:
        raise ValueError("Capacity planning only shrinks the observed initialized pool")
    ratio = blocks / concurrency
    per_request = round(ratio)
    if per_request < 1 or not math.isclose(ratio, per_request, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("Cannot infer an integer device-pool block budget per request")
    return dict(
        status="planned",
        scope="Logical single-engine budget inferred from initialized metrics; verify again after launch",
        model_length=model_length,
        target_tokens=target_tokens,
        observed_tokens=reported_tokens,
        observed_num_gpu_blocks=blocks,
        observed_max_concurrency=concurrency,
        blocks_per_full_request=per_request,
        num_gpu_blocks_override=per_request * (target_tokens // model_length),
        cache_geometry={
            key: fields.get(key)
            for key in ("block_size", "mamba_block_size", "mamba_cache_mode", "mamba_ssm_cache_dtype")
        },
    )
