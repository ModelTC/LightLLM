import torch


class DenseKVWriter:
    def __init__(self, view):
        self.view = view
        self.k, self.v = view.fields

    def __call__(self, *, token_indices, kv):
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv

        heads, dim = self.k.shape[1:]
        if kv.shape != (token_indices.numel(), 2 * heads, dim) or kv.dtype != self.k.dtype:
            raise ValueError("KV writer input does not match its bound format")
        if token_indices.device != self.k.device or kv.device != self.k.device:
            raise ValueError("KV writer indices and payload must reside on the bound device")
        if token_indices.numel():
            destindex_copy_kv(K=kv[:, :heads], DestLoc=token_indices, Out=self.k)
            destindex_copy_kv(K=kv[:, heads:], DestLoc=token_indices, Out=self.v)
