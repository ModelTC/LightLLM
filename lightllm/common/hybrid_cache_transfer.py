import torch
import triton

from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size


class HybridCpuCacheTransfer:
    def __init__(self, kv_buffer, linear_config, big_page_buffers):
        self.kv_buffer = kv_buffer
        self.linear_config = linear_config
        self.big_page_buffers = big_page_buffers

    def get_checkpoint_lens(self, page_indexes, cpu_cache_client):
        cache = cpu_cache_client.cpu_kv_cache_tensor
        metadata = cache.view(cache.shape[0], -1).view(torch.uint8)[:, -16:].view(torch.int64).numpy()
        return [int(length) if version == 1 and length >= 0 else None for length, version in metadata[page_indexes]]

    def load(self, mem_indexes, page_indexes, cpu_cache_client, big_page_buffer_ids_cpu):
        args = get_env_start_args()
        assert mem_indexes.is_cuda and page_indexes.is_cuda
        # 碎页情况的处理
        has_tail_page = len(mem_indexes) % args.cpu_cache_token_page_size != 0
        if has_tail_page:
            padded_token_num = triton.cdiv(
                len(mem_indexes), args.cpu_cache_token_page_size
            ) * args.cpu_cache_token_page_size - len(mem_indexes)
            mem_indexes = torch.nn.functional.pad(mem_indexes, (0, padded_token_num), mode="constant", value=-1)

        big_page_buffer_ids_gpu = torch.tensor(big_page_buffer_ids_cpu, dtype=torch.int64, device="cpu").cuda(
            non_blocking=True
        )

        assert len(big_page_buffer_ids_gpu) == len(page_indexes)

        from lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy import (
            copy_cpu_cache_to_kv_buffer,
        )

        copy_cpu_cache_to_kv_buffer(
            mem_indexes=mem_indexes,
            big_page_buffer_ids=big_page_buffer_ids_gpu,
            page_indexes=page_indexes,
            gpu_full_att_kv_state=self.kv_buffer,
            cpu_kv_conv_state=self.big_page_buffers.conv_state_cache.buffer,
            cpu_kv_ssm_state=self.big_page_buffers.ssm_state_cache.buffer,
            cpu_cache_tensor=cpu_cache_client.cpu_kv_cache_tensor,
            tp_rank=get_current_rank_in_dp(),
            tp_world_size=get_dp_world_size(),
            big_page_token_num=args.cpu_cache_token_page_size,
            linear_config=self.linear_config,
        )

    def offload(
        self,
        mem_indexes,
        page_indexes,
        page_readies,
        cpu_cache_client,
        big_page_buffer_ids_cpu,
        tail_state,
        page_indexes_cpu=None,
        page_readies_cpu=None,
    ):
        args = get_env_start_args()
        if not hasattr(self, "big_page_ids_buffer_store"):
            self.big_page_ids_buffer_store = torch.empty((1024 * 1024 * 4,), dtype=torch.int64, device="cuda")
            # 多申请3个cpu cache token page size，用于处理碎页情况，碎页情况需要将对应的大页数据拷贝到临时的大页上，
            # 再从大页上拷贝到对应的运行态页面上
            self.mem_indexes_buffer = torch.empty(
                (args.max_req_total_len + 3 * args.cpu_cache_token_page_size,), dtype=torch.int32, device="cuda"
            )

        assert mem_indexes.is_cuda and page_indexes.is_cuda and page_readies.is_cuda

        assert len(mem_indexes) % args.linear_att_hash_page_size == 0
        assert triton.cdiv(len(mem_indexes), args.cpu_cache_token_page_size) == len(page_indexes)

        if len(mem_indexes) % args.cpu_cache_token_page_size != 0:
            # 存在不满大页的碎页的页面存在需要复制的情况
            dst_len = triton.cdiv(len(mem_indexes), args.cpu_cache_token_page_size) * args.cpu_cache_token_page_size
            assert dst_len <= self.mem_indexes_buffer.shape[0]
            dst_mem_indexes = self.mem_indexes_buffer[0:dst_len].fill_(-1)
            dst_mem_indexes[0 : len(mem_indexes)].copy_(mem_indexes, non_blocking=True)
            mem_indexes = dst_mem_indexes
            from lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy import (
                copy_linear_att_state_to_linear_att_state,
            )

            src_conv_state, src_ssm_state = tail_state
            dst_conv_state, dst_ssm_state = self.big_page_buffers.get_state_cache(
                buffer_idx=big_page_buffer_ids_cpu[-1],
            )
            copy_linear_att_state_to_linear_att_state(
                src_conv_state=src_conv_state,
                src_ssm_state=src_ssm_state,
                dst_conv_state=dst_conv_state,
                dst_ssm_state=dst_ssm_state,
            )

        assert len(big_page_buffer_ids_cpu) == len(page_indexes) == len(page_readies)

        if get_current_rank_in_dp() == 0:
            if page_indexes_cpu is None:
                page_indexes_cpu = page_indexes.cpu().tolist()
            if page_readies_cpu is None:
                page_readies_cpu = page_readies.cpu().tolist()
            cache = cpu_cache_client.cpu_kv_cache_tensor
            metadata = cache.view(cache.shape[0], -1).view(torch.uint8)[:, -16:].view(torch.int64).numpy()
            for page_index, ready, slot in zip(
                page_indexes_cpu, page_readies_cpu, big_page_buffer_ids_cpu, strict=True
            ):
                if page_index != -1 and not ready:
                    length = self.big_page_buffers.checkpoint_lens[slot]
                    metadata[page_index] = (-1 if length is None else length, 1)

        big_page_buffer_ids_cpu = torch.tensor(
            big_page_buffer_ids_cpu, dtype=torch.int64, device="cpu", pin_memory=True
        )
        assert len(big_page_buffer_ids_cpu) <= self.big_page_ids_buffer_store.shape[0]
        big_page_buffer_ids_gpu = self.big_page_ids_buffer_store[0 : len(big_page_buffer_ids_cpu)]
        big_page_buffer_ids_gpu.copy_(big_page_buffer_ids_cpu, non_blocking=True)

        from lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy import (
            copy_kv_buffer_to_cpu_cache,
        )

        copy_kv_buffer_to_cpu_cache(
            mem_indexes=mem_indexes,
            page_indexes=page_indexes,
            page_readies=page_readies,
            big_page_buffer_ids=big_page_buffer_ids_gpu,
            gpu_kv_full_att_state=self.kv_buffer,
            cpu_kv_conv_state=self.big_page_buffers.conv_state_cache.buffer,
            cpu_kv_ssm_state=self.big_page_buffers.ssm_state_cache.buffer,
            cpu_cache_tensor=cpu_cache_client.cpu_kv_cache_tensor,
            tp_rank=get_current_rank_in_dp(),
            tp_world_size=get_dp_world_size(),
            big_page_token_num=args.cpu_cache_token_page_size,
            linear_config=self.linear_config,
        )
        return
