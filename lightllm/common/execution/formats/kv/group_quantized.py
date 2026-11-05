import torch


class GroupQuantizedKVWriter:
    def __init__(self, view):
        self.k, self.v, self.k_scale, self.v_scale = view.fields
        self.group_size = view.format.group_size
        self.bits = 4 if view.format.encoding == "int4" else 8
        self.head_dim = view.state.params.head_dim

    def __call__(self, *, token_indices, kv):
        heads = self.k.shape[1]
        if kv.shape != (token_indices.numel(), 2 * heads, self.head_dim):
            raise ValueError("KV writer input does not match its bound logical shape")
        if kv.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("grouped KV writer requires FP16/BF16 input")
        if token_indices.device != self.k.device or kv.device != self.k.device:
            raise ValueError("KV writer indices and payload must reside on the bound device")
        if not token_indices.numel():
            return
        for offset, data, scale in ((0, self.k, self.k_scale), (heads, self.v, self.v_scale)):
            if self.bits == 4:
                from lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int4kv_copy_kv import destindex_copy_int4kv

                destindex_copy_int4kv(kv[:, offset : offset + heads], token_indices, data, scale, self.group_size)
            else:
                from lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int8kv_copy_kv import (
                    destindex_copy_quantize_kv,
                )

                destindex_copy_quantize_kv(kv[:, offset : offset + heads], token_indices, data, scale, self.group_size)
