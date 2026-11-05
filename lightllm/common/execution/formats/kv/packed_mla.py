import torch


class PackedMLAKVWriter:
    def __init__(self, view):
        self.cache = view.fields[0]
        self.quantized = view.format.encoding == "fp8_group128"

    def __call__(self, *, token_indices, kv):
        if kv.shape != (token_indices.numel(), 1, 576) or kv.dtype != torch.bfloat16:
            raise ValueError("packed MLA writer requires BF16 latent 512 and rotary 64")
        if kv.device != self.cache.device or token_indices.device != self.cache.device:
            raise ValueError("packed MLA inputs must reside on the bound device")
        if not token_indices.numel():
            return
        token_indices = token_indices.contiguous()
        if self.quantized:
            from lightllm.common.basemodel.triton_kernel.kv_copy.destindex_copy_kv_flashmla_fp8 import (
                destindex_copy_kv_flashmla_fp8,
            )

            destindex_copy_kv_flashmla_fp8(
                kv[..., :512],
                kv[..., 512:],
                token_indices,
                self.cache[..., :512].view(torch.float8_e4m3fn),
                self.cache[..., 512:528].view(torch.float32),
                self.cache[..., 528:656].view(torch.bfloat16),
            )
        else:
            from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv

            destindex_copy_kv(K=kv, DestLoc=token_indices, Out=self.cache)
