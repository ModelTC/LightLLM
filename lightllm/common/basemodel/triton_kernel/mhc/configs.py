# SPDX-License-Identifier: Apache-2.0

"""Launch configurations for the shared mHC kernels; arithmetic precision is fixed."""


def projection_configs():
    return [{"n_splits": splits} for splits in [0, 1, 2, 4, 8, 16, 32, 64]]


def projection_key(x, fn):
    return {"K": x.shape[1], "M": fn.shape[0], "x_dtype": str(x.dtype), "fn_dtype": str(fn.dtype)}


def pre_norm_configs():
    return [{"num_warps": warps, "unroll_sinkhorn": unroll} for warps in [1, 2, 4, 8, 16] for unroll in [True, False]]


def pre_norm_key(x, gemm_partial, norm_weight, streams, sinkhorn_iters):
    return {
        "hidden": x.shape[1] // streams,
        "streams": streams,
        "n_splits": gemm_partial.shape[0],
        "sinkhorn_iters": sinkhorn_iters,
        "x_dtype": str(x.dtype),
        "norm_dtype": str(norm_weight.dtype),
    }


def post_projection_configs():
    return [
        {"n_splits": splits, "BLOCK_M": block_m, "num_warps": warps}
        for splits in [4, 8, 16, 32]
        for block_m in [1, 2, 4, 8]
        for warps in [2, 4, 8]
    ]


def post_projection_key(layer_output, fn):
    return {"hidden": layer_output.shape[1], "x_dtype": str(layer_output.dtype), "fn_dtype": str(fn.dtype)}


def decode_post_pre_configs():
    return [dict(config, norm_num_warps=warps) for config in post_projection_configs() for warps in [4, 8]]


def decode_post_pre_key(layer_output, fn, norm_weight, sinkhorn_iters):
    return dict(post_projection_key(layer_output, fn), norm_dtype=str(norm_weight.dtype), sinkhorn_iters=sinkhorn_iters)


def post_configs():
    return [
        {"BLOCK_H": block_h, "num_warps": warps}
        for block_h in [256, 512, 1024, 2048, 4096]
        for warps in [2, 4, 8]
        if block_h < 4096 or warps >= 4
    ]


def post_key(layer_output, residual_mix, post_mix, streams):
    return {
        "hidden": layer_output.shape[1],
        "streams": streams,
        "x_dtype": str(layer_output.dtype),
        "residual_mix_dtype": str(residual_mix.dtype),
        "post_mix_dtype": str(post_mix.dtype),
    }


def head_configs():
    return [{"BLOCK_K": block_k, "num_warps": warps} for block_k in [512, 1024, 2048, 4096, 8192] for warps in [4, 8]]


def head_key(x, fn, streams):
    return {
        "hidden": x.shape[1] // streams,
        "streams": streams,
        "x_dtype": str(x.dtype),
        "fn_dtype": str(fn.dtype),
    }
