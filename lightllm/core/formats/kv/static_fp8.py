import torch


class StaticFP8KVWriter:
    def __init__(self, view):
        self.k, self.v, self.scales, _ = view.fields
        self.per_tensor = view.format.encoding == "fp8_per_tensor"

    def __call__(self, *, token_indices, kv):
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv_fp8 import destindex_copy_kv_fp8

        heads, dim = self.k.shape[1:]
        if kv.shape != (token_indices.numel(), heads * 2, dim) or kv.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("FP8 writer input does not match its bound logical shape")
        if kv.device != self.k.device or token_indices.device != self.k.device:
            raise ValueError("FP8 writer inputs must reside on the bound device")
        if token_indices.numel():
            destindex_copy_kv_fp8(kv, token_indices, self.scales, self.k, self.per_tensor, Out_v=self.v)
