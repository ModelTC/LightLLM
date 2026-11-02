from functools import partial

import torch


def _cuda_writer(run, storage):
    def write(*, token_indices, kv):
        assert token_indices.is_cuda, "token indices must reside on CUDA"
        if not token_indices.numel():
            return
        assert storage.is_cuda, "KV writes require CUDA storage"
        run(kv, token_indices)

    return write


def bind_plain_kv_writer(k, v):
    from ..triton_kernel.destindex_copy_kv import destindex_copy_kv_pair

    return _cuda_writer(partial(destindex_copy_kv_pair, k=k, v=v), k)


def bind_latent_writer(cache):
    from ..triton_kernel.destindex_copy_kv import destindex_copy_kv

    return _cuda_writer(partial(destindex_copy_kv, Out=cache), cache)


def bind_group_quantized_writer(k, v, bits, group_size):
    if bits == 4:
        from ..triton_kernel.kv_copy.ppl_int4kv_copy_kv import destindex_copy_int4kv

        def bind(field):
            return partial(
                destindex_copy_int4kv, KV_buffer=field[0], KV_scale_buffer=field[1], quant_group_size=group_size
            )

    elif bits == 8:
        from ..triton_kernel.kv_copy.ppl_int8kv_copy_kv import destindex_copy_quantize_kv

        def bind(field):
            return partial(destindex_copy_quantize_kv, Out=field[0], Out_scale=field[1], quant_group_dim=group_size)

    else:
        raise ValueError(f"unsupported grouped KV bit width {bits}")
    write_k, write_v = bind(k), bind(v)
    heads = k[0].shape[1]

    def run(kv, indices):
        write_k(kv[:, :heads], indices)
        write_v(kv[:, heads : 2 * heads], indices)

    return _cuda_writer(run, k[0])


def bind_static_fp8_writer(k, v, scales, per_tensor):
    from ..triton_kernel.destindex_copy_kv_fp8 import destindex_copy_kv_fp8

    return _cuda_writer(
        partial(
            destindex_copy_kv_fp8,
            scales=scales,
            Out=k.view(torch.float8_e4m3fn),
            Out_v=v.view(torch.float8_e4m3fn),
            is_per_tensor_quant=per_tensor,
        ),
        k,
    )


def bind_packed_fp8_writer(cache, nope_dim, rope_dim, group_size):
    from ..triton_kernel.kv_copy.destindex_copy_kv_flashmla_fp8 import destindex_copy_kv_flashmla_fp8

    rope_start = nope_dim + nope_dim // group_size * torch.float32.itemsize
    run_kernel = partial(
        destindex_copy_kv_flashmla_fp8,
        O_nope=cache[..., :nope_dim].view(torch.float8_e4m3fn),
        O_scale=cache[..., nope_dim:rope_start].view(torch.float32),
        O_rope=cache[..., rope_start : rope_start + rope_dim * torch.bfloat16.itemsize].view(torch.bfloat16),
    )

    def run(kv, indices):
        run_kernel(KV_nope=kv[..., :nope_dim], KV_rope=kv[..., nope_dim:], DestLoc=indices)

    return _cuda_writer(run, cache)


def bind_e8m0_writer(cache):
    from lightllm.common.attention_state.full_att.packed_mla.fp8_e8m0_packed_mla_write import write_fp8_e8m0_packed_kv

    return _cuda_writer(partial(write_fp8_e8m0_packed_kv, packed_kv=cache), cache)


def fused_compressor_write_only(*, token_indices, kv):
    raise NotImplementedError("compressed KV is written by the fused compressor")
