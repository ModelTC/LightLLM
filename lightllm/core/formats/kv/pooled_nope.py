import torch


class PooledNoPEKVWriter:
    def __init__(self, view):
        self.cache = view.fields[0]
        self.dtype = view.format.field_dtypes[0] if view.format.encoding == "plain" else torch.bfloat16
        self.quantized = view.format.encoding == "fp8_group64_e8m0"

    def __call__(self, *, token_indices, kv):
        if kv.shape != (token_indices.numel(), 1, 512) or kv.dtype != self.dtype:
            raise ValueError("pooled NoPE writer requires the declared 512-dimensional latent")
        if kv.device != self.cache.device or token_indices.device != self.cache.device:
            raise ValueError("pooled NoPE writes must use the bound device")
        if not token_indices.numel():
            return
        if self.quantized:
            from lightllm.common.attention_state.full_att.packed_mla.fp8_e8m0_packed_mla_write import (
                write_fp8_e8m0_packed_kv,
            )

            write_fp8_e8m0_packed_kv(kv, token_indices, self.cache)
        else:
            from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv

            destindex_copy_kv(K=kv, DestLoc=token_indices.contiguous(), Out=self.cache)
