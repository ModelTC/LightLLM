import torch


class MLAKVWriter:
    def __init__(self, view):
        self.latent = view.fields[0]

    def __call__(self, *, token_indices, kv):
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv

        if kv.shape != (token_indices.numel(), *self.latent.shape[1:]) or kv.dtype != self.latent.dtype:
            raise ValueError("MLA writer input does not match its bound latent format")
        if kv.device != self.latent.device or token_indices.device != self.latent.device:
            raise ValueError("MLA writer inputs must reside on the bound device")
        if token_indices.numel():
            destindex_copy_kv(K=kv, DestLoc=token_indices, Out=self.latent)
