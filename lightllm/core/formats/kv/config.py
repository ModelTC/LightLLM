from lightllm.core.contracts.state import FormatSpec
from lightllm.core.compute.attention.mla.spec import MLAAttentionSpec
from lightllm.core.compute.attention.sparse_mla.spec import SparseMLAAttentionSpec
from lightllm.core.compute.attention.pooled_nope.spec import PooledNoPEAttentionSpec


def history_formats(specs, *, kv_type, model_dtype, page_size, shard, group_size, calibration=None, capacity_tokens=1):
    import torch

    mla = any(isinstance(spec, MLAAttentionSpec) for spec in specs.values())
    sparse = any(isinstance(spec, SparseMLAAttentionSpec) for spec in specs.values())
    pooled = any(isinstance(spec, PooledNoPEAttentionSpec) for spec in specs.values())
    encodings = {
        "None": ("plain", model_dtype, ("k", "v"), None),
        "int4kv": ("int4", torch.int8, ("k", "v", "k_scale", "v_scale"), model_dtype),
        "int8kv": ("int8", torch.int8, ("k", "v", "k_scale", "v_scale"), model_dtype),
        "fp8kv_sph": ("fp8_per_head", torch.float8_e4m3fn, ("k", "v", "kv_scale", "q_scale"), torch.float32),
        "fp8kv_spt": ("fp8_per_tensor", torch.float8_e4m3fn, ("k", "v", "kv_scale", "q_scale"), torch.float32),
        "fp8kv_dsa": ("fp8_group128", torch.uint8, ("latent", "indexer"), torch.float32),
    }
    encoding, dtype, fields, scale_dtype = encodings[kv_type]
    format = FormatSpec(
        encoding,
        dtype,
        "token_latent" if mla else "token_head_dim",
        ("latent",) if mla else fields,
        page_size,
        shard,
        calibration=calibration,
        group_size=group_size if encoding in ("int4", "int8") else None,
        scale_dtype=scale_dtype,
    )
    if sparse:
        format = FormatSpec(
            encoding,
            dtype,
            "token_packed_mla",
            ("latent", "indexer"),
            page_size,
            shard,
            group_size=128 if encoding == "fp8_group128" else None,
            scale_dtype=scale_dtype,
            field_dtypes=(dtype, torch.uint8),
        )
    window_format = None
    if pooled:
        quantized = kv_type == "fp8kv_dsa"
        dtype = torch.uint8 if quantized else model_dtype
        format = FormatSpec(
            "fp8_group64_e8m0" if quantized else "plain",
            dtype,
            "token_pooled_nope",
            ("latent", "indexer"),
            page_size,
            shard,
            group_size=64 if quantized else None,
            scale_dtype=torch.uint8 if quantized else None,
            field_dtypes=(dtype, torch.uint8),
        )
        window_format = FormatSpec(
            "plain",
            model_dtype,
            "request_indexer_window",
            ("tail",),
            1,
            shard,
            candidate_slots=capacity_tokens,
        )
    return format, window_format
