from typing import TYPE_CHECKING, List

import torch

from lightllm.common.state_cache_manager import LayerCache, LinearAttCacheConfig, LinearAttCacheManager
from lightllm.utils.envs_utils import get_env_start_args

from .hybrid_base import HybridAttentionReqManager


if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq


class ReqManagerForMamba(HybridAttentionReqManager):
    def __init__(
        self,
        max_request_num,
        max_sequence_length,
        mem_manager,
        linear_config: LinearAttCacheConfig,
        recurrent_kind="gdn",
    ):
        super().__init__(max_request_num, max_sequence_length, mem_manager)
        args = get_env_start_args()
        self.mtp_step = args.mtp_step
        replay = getattr(args, "enable_replayssm", False)
        if (
            replay
            and self.mtp_step == 0
            and (recurrent_kind == "kda" or linear_config.ssm_state_dtype != torch.float32)
        ):
            from lightllm.utils.log_utils import init_logger

            init_logger(__name__).info("ReplaySSM: non-speculative BF16/KDA decode uses the recurrent kernel")
            replay = False
        self.ssm_slots_per_req = 1 if replay else self.mtp_step + 1
        self.replay_cache = None
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
        if replay:
            from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import ReplaySSMCache

            if recurrent_kind == "gdn" and linear_config.ssm_state_dtype == torch.float32:
                self.replay_cache = ReplaySSMCache(
                    self.req_to_ssm_state.buffer,
                    args.replayssm_cache_len,
                    self.mtp_step + 1,
                    linear_config.conv_state_dtype,
                )
            else:
                from lightllm.common.basemodel.triton_kernel.linear_att.replayssm_compact import CompactSSMCache

                self.replay_cache = CompactSSMCache(
                    self.req_to_ssm_state.buffer, self.mtp_step + 1, linear_config.conv_state_dtype, kind=recurrent_kind
                )
        return

    def init_hybrid_attention_state(self, req: "InferReq"):
        if self.replay_cache is not None:
            self.replay_cache.reset(req.req_idx)
        conv_index = req.req_idx
        ssm_start = req.req_idx * self.ssm_slots_per_req
        self.req_to_conv_state.buffer[:, conv_index, ...].fill_(0)
        # #17: zero the FULL (mtp_step + 1)-row SSM block, not just canonical row +0, so a future
        # first-step verify reading offset>0 after fresh init never hits a never-written row (NaN).
        self.req_to_ssm_state.buffer[:, ssm_start : ssm_start + self.ssm_slots_per_req, ...].fill_(0)
        if self.req_to_mtp_state_index is not None:
            self.req_to_mtp_state_index[req.req_idx] = 0
        return

    def create_small_page_cache_manager(self, size: int):
        self.small_page_buffers = LinearAttCacheManager(size=size, linear_config=self.linear_config)
        return self.small_page_buffers

    def save_big_page_states(self, b_req_idx: torch.Tensor, req_indexes: List[int], buffer_indexes: List[int]):
        from lightllm.common.basemodel.triton_kernel.linear_att_copy import copy_linear_att_state_to_kv_buffer

        if self.replay_cache is not None:
            self.replay_cache.materialize(b_req_idx)
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
            conv_offsets=self.req_to_mtp_state_index if self.replay_cache is not None else None,
        )
        return

    def save_state(self, req_idx: int, buffer_idx: int, state_cache_manager: LinearAttCacheManager):
        if self.replay_cache is not None:
            self.replay_cache.materialize(torch.tensor([req_idx], dtype=torch.int32, device="cuda"))
        # checkpoint 只保存标准 conv 窗口和请求的基准 SSM 状态，不包含 MTP 扩展运行态。
        conv_cache_width = self.linear_config.get_conv_state_shape()[-1]
        gpu_conv_state = self.req_to_conv_state.buffer[:, req_idx, ..., :conv_cache_width]
        if self.replay_cache is not None and self.req_to_mtp_state_index is not None:
            offsets = torch.arange(conv_cache_width, device="cuda") + self.req_to_mtp_state_index[req_idx]
            gpu_conv_state = self.req_to_conv_state.buffer[:, req_idx].index_select(-1, offsets)
        gpu_ssm_state = self.req_to_ssm_state.buffer[:, req_idx * self.ssm_slots_per_req, ...]
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

    def update_mtp_state(self, b_req_mtp_start_loc, b_req_idx, b_mtp_index, accepted_index, verify_width):
        from lightllm.common.basemodel.triton_kernel.mtp_utils import linear_att_mtp_state_index_update

        linear_att_mtp_state_index_update(
            req_to_mtp_state_index=self.req_to_mtp_state_index,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            b_req_idx=b_req_idx,
            b_mtp_index=b_mtp_index,
            accepted_index=accepted_index,
            verify_width=verify_width,
        )

        if self.replay_cache is not None:
            reqs = b_req_idx[b_req_mtp_start_loc.long()]
            self.replay_cache.commit(reqs, self.req_to_mtp_state_index)

    def restore_state(self, req: "InferReq", state_cache_manager: LinearAttCacheManager, buffer_idx: int):
        if self.replay_cache is not None:
            self.replay_cache.reset(req.req_idx)
        conv_state, ssm_state = state_cache_manager.get_state_cache(buffer_idx=buffer_idx)
        conv_dest = req.req_idx
        ssm_dest = req.req_idx * self.ssm_slots_per_req
        conv_cache_width = conv_state.shape[-1]
        self.req_to_conv_state.buffer[:, conv_dest, ..., :conv_cache_width] = conv_state
        self.req_to_ssm_state.buffer[:, ssm_dest, ...] = ssm_state
        if self.req_to_mtp_state_index is not None:
            self.req_to_mtp_state_index[req.req_idx] = 0
        return
