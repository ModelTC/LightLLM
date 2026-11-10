"""GQA 的物理写入/复制实现；层映射、容量和 token 生命周期仍由 State 管理。"""

import torch

from lightllm.platform import Platform


class TritonKVOperations:
    @staticmethod
    def write(key, value, indices, kv):
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv

        heads = key.shape[1]
        destindex_copy_kv(K=kv[:, :heads], DestLoc=indices, Out=key)
        destindex_copy_kv(K=kv[:, heads:], DestLoc=indices, Out=value)

    @staticmethod
    def copy(key, value, src, dst):
        from lightllm.common.basemodel.triton_kernel.kv_move import copy_kv_buffer_to_kv_buffer

        copy_kv_buffer_to_kv_buffer(src_mem_index=src, dst_mem_index=dst, kv_buffer=key)
        copy_kv_buffer_to_kv_buffer(src_mem_index=src, dst_mem_index=dst, kv_buffer=value)


class TorchKVOperations:
    @staticmethod
    def write(key, value, indices, kv):
        indices = indices.to(torch.int64)
        key.index_copy_(0, indices, kv[:, : key.shape[1]])
        value.index_copy_(0, indices, kv[:, key.shape[1] :])

    @staticmethod
    def copy(key, value, src, dst):
        src, dst = src.to(torch.int64), dst.to(torch.int64)
        key.index_copy_(1, dst, key.index_select(1, src))
        value.index_copy_(1, dst, value.index_select(1, src))


def get_kv_operations(runtime):
    if runtime.platform is Platform.CUDA:
        return TritonKVOperations
    if runtime.platform in (Platform.CPU, Platform.MUSA, Platform.ASCEND):
        return TorchKVOperations
    raise ValueError(f"GQA KV writes are not implemented on {runtime.platform.value}")
