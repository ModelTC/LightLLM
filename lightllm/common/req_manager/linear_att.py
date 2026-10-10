from typing import TYPE_CHECKING, List

import torch

from lightllm.common.state_cache_manager import LayerCache, LinearAttCacheConfig, LinearAttCacheManager
from lightllm.utils.envs_utils import get_env_start_args

from .hybrid_base import HybridAttentionReqManager


if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq


class ReqManagerForMamba(HybridAttentionReqManager):
    kda = False

    def __init__(
        self,
        max_request_num,
        max_sequence_length,
        mem_manager,
        linear_config: LinearAttCacheConfig,
    ):
        super().__init__(max_request_num, max_sequence_length, mem_manager)
        args = get_env_start_args()
        self.mtp_step = args.mtp_step
        self.ssm_slots_per_req = self.mtp_step + 1 if args.ssm_state_mode == "native" else 1
        self.ssm_update_cache = None
        # 因为在mtp的推理中，需要标记每个请求对应的mtp index状态(conv state 和 ssm state)，在mtp对应序列中
        # 的真实位置，所以需要需要一个标记来记录，不然算子无法找到真实的处理起点。
        self.req_to_mtp_state_index = (
            torch.zeros((max_request_num + 1,), dtype=torch.int32, device="cuda") if self.mtp_step > 0 else None
        )
        # 突然想到， 在linear att 开启mtp的模式中，现在的prefill linear att 算子默认是从0的位置读取信息进行操作
        # 所以不能支持 prefill decode mixed 操作了，因为一个decode过的请求，重新用prefill 算子跑，会出现读错linear
        # 状态位置的问题。导致bug, 在这里加个断言，以后可以支持上 TODO
        if self.mtp_step > 0:
            assert get_env_start_args().enable_prefill_decode_mixed is False

        self.big_page_token_num = (
            get_env_start_args().linear_att_page_block_num * get_env_start_args().linear_att_hash_page_size
        )
        self.linear_config = linear_config

        self.req_to_conv_state = LayerCache(
            size=(max_request_num + 1),
            dtype=self.linear_config.conv_state_dtype,
            shape=self.linear_config.get_mtp_conv_state_shape(mtp_step=self.mtp_step),
            layer_num=self.linear_config.linear_layer_num,
            device="cuda",
        )
        self.req_to_ssm_state = LayerCache(
            size=(max_request_num + 1) * self.ssm_slots_per_req,
            dtype=self.linear_config.ssm_state_dtype,
            shape=self.linear_config.get_ssm_state_shape(),
            layer_num=self.linear_config.linear_layer_num,
            device="cuda",
        )
        if args.ssm_state_mode == "replay":
            from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import ReplaySSMCache

            self.ssm_update_cache = ReplaySSMCache(
                self.req_to_ssm_state.buffer,
                args.replayssm_cache_len,
                self.mtp_step + 1,
                linear_config.conv_state_dtype,
                num_key_heads=linear_config.num_linear_k_heads,
                kda=self.kda,
                projection_mode=args.replayssm_projection_mode,
            )
        elif args.ssm_state_mode == "compact":
            from lightllm.common.basemodel.triton_kernel.linear_att.replayssm_compact import CompactSSMCache

            self.ssm_update_cache = CompactSSMCache(
                self.req_to_ssm_state.buffer,
                self.mtp_step + 1,
                linear_config.conv_state_dtype,
                num_key_heads=linear_config.num_linear_k_heads,
                kda=self.kda,
            )
        return

    def init_hybrid_attention_state(self, req: "InferReq"):
        if self.ssm_update_cache is not None:
            # 清除这个请求槽位之前的历史记录标记
            self.ssm_update_cache.clear_history(req.req_idx)
        conv_index = req.req_idx
        ssm_start = req.req_idx * self.ssm_slots_per_req
        self.req_to_conv_state.buffer[:, conv_index, ...].fill_(0)
        self.req_to_ssm_state.buffer[:, ssm_start : ssm_start + self.ssm_slots_per_req, ...].fill_(0)
        if self.req_to_mtp_state_index is not None:
            self.req_to_mtp_state_index[req.req_idx].zero_()
        return

    def create_small_page_cache_manager(self, size: int):
        self.small_page_buffers = LinearAttCacheManager(size=size, linear_config=self.linear_config)
        return self.small_page_buffers

    def save_big_page_states(self, b_req_idx: torch.Tensor, req_indexes: List[int], buffer_indexes: List[int]):
        from lightllm.common.basemodel.triton_kernel.linear_att_copy import copy_linear_att_state_to_kv_buffer

        if self.ssm_update_cache is not None:
            return super().save_big_page_states(b_req_idx, req_indexes, buffer_indexes)
        buffer_indexes = torch.tensor(buffer_indexes, dtype=torch.int32, device="cpu").cuda(non_blocking=True)
        state_cache_manager = self.big_page_buffers
        copy_linear_att_state_to_kv_buffer(
            b_req_idx=b_req_idx,
            big_page_buffer_ids=buffer_indexes,
            gpu_conv_state=self.req_to_conv_state.buffer,
            gpu_ssm_state=self.req_to_ssm_state.buffer,
            cpu_kv_conv_state=state_cache_manager.conv_state_cache.buffer,
            cpu_kv_ssm_state=state_cache_manager.ssm_state_cache.buffer,
            mtp_step=self.ssm_slots_per_req - 1,
            conv_offsets=self.req_to_mtp_state_index,
        )
        return

    def save_state(self, req_idx: int, buffer_idx: int, state_cache_manager: LinearAttCacheManager):
        # checkpoint 只保存标准 conv 窗口和请求的基准 SSM 状态，不包含 MTP 扩展运行态。
        conv_cache_width = self.linear_config.get_conv_state_shape()[-1]
        gpu_conv_state = self.req_to_conv_state.buffer[:, req_idx, ..., :conv_cache_width]
        if self.req_to_mtp_state_index is not None:
            offsets = torch.arange(conv_cache_width, device="cuda") + self.req_to_mtp_state_index[req_idx]
            gpu_conv_state = self.req_to_conv_state.buffer[:, req_idx].index_select(-1, offsets)
        if self.ssm_update_cache is not None:
            gpu_ssm_state = self.ssm_update_cache.snapshot_accepted_state(req_idx)
        else:
            gpu_ssm_state = self.req_to_ssm_state.buffer[:, req_idx * self.ssm_slots_per_req, ...]
        if self.ssm_slots_per_req > 1:
            state_index = req_idx * self.ssm_slots_per_req + self.req_to_mtp_state_index[req_idx]
            gpu_ssm_state = self.req_to_ssm_state.buffer.index_select(1, state_index.long().view(1))[:, 0]
        dst_conv_state, dst_ssm_state = state_cache_manager.get_state_cache(buffer_idx=buffer_idx)
        dst_conv_state.copy_(gpu_conv_state, non_blocking=True)
        dst_ssm_state.copy_(gpu_ssm_state, non_blocking=True)

    def get_mamba_cache(self, layer_idx_in_all: int):
        assert (
            0 <= layer_idx_in_all < self.linear_config.all_layer_num
        ), f"invalid transformer layer index {layer_idx_in_all}"
        layer_idx_in_linear = layer_idx_in_all - (layer_idx_in_all // self.linear_config.full_attention_interval)
        conv_states = self.req_to_conv_state.buffer[layer_idx_in_linear]
        ssm_states = self.req_to_ssm_state.buffer[layer_idx_in_linear]
        return conv_states, ssm_states

    def update_mtp_state(
        self, b_req_mtp_start_loc, b_req_idx, b_mtp_index, accepted_index, verify_width, ssm_accept_batches=None
    ):
        from lightllm.common.basemodel.triton_kernel.mtp_utils import linear_att_mtp_state_index_update

        replay_cursors = None
        if self.ssm_update_cache is not None:
            from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import ReplaySSMCache

            if isinstance(self.ssm_update_cache, ReplaySSMCache) and self.ssm_update_cache.verify_width > 1:
                replay_cursors = self.ssm_update_cache.cursors

        linear_att_mtp_state_index_update(
            req_to_mtp_state_index=self.req_to_mtp_state_index,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            b_req_idx=b_req_idx,
            b_mtp_index=b_mtp_index,
            accepted_index=accepted_index,
            verify_width=verify_width,
            replay_cursors=replay_cursors,
        )

        if self.ssm_update_cache is not None and replay_cursors is None:
            reqs = b_req_idx[b_req_mtp_start_loc.long()]
            if ssm_accept_batches is None or isinstance(self.ssm_update_cache, ReplaySSMCache):
                self.ssm_update_cache.accept_updates(reqs, self.req_to_mtp_state_index)
            else:
                assert sum(size for size, _ in ssm_accept_batches) == reqs.numel()
                start = 0
                for size, config in ssm_accept_batches:
                    if size:
                        self.ssm_update_cache.accept_updates(
                            reqs[start : start + size], self.req_to_mtp_state_index, run_config=config
                        )
                    start += size

    def restore_state(self, req: "InferReq", state_cache_manager: LinearAttCacheManager, buffer_idx: int):
        if self.ssm_update_cache is not None:
            self.ssm_update_cache.clear_history(req.req_idx)
        conv_state, ssm_state = state_cache_manager.get_state_cache(buffer_idx=buffer_idx)
        conv_dest = req.req_idx
        ssm_dest = req.req_idx * self.ssm_slots_per_req
        conv_cache_width = conv_state.shape[-1]
        self.req_to_conv_state.buffer[:, conv_dest, ..., :conv_cache_width] = conv_state
        self.req_to_ssm_state.buffer[:, ssm_dest, ...] = ssm_state
        if self.req_to_mtp_state_index is not None:
            self.req_to_mtp_state_index[req.req_idx].zero_()
        return
