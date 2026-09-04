from typing import TYPE_CHECKING

import torch

from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig
from lightllm.common.linear_att_cache_manager.layer_cache import LayerCache
from lightllm.common.linear_att_cache_manager.linear_att_buffer_manager import LinearAttCacheManager
from lightllm.utils.envs_utils import get_env_start_args

from .base import ReqManager


if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq


class ReqManagerForMamba(ReqManager):
    def __init__(self, max_request_num, max_sequence_length, mem_manager, linear_config: LinearAttCacheConfig):
        super().__init__(max_request_num, max_sequence_length, mem_manager)
        self.mtp_step = get_env_start_args().mtp_step
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
            size=(max_request_num + 1) * (self.mtp_step + 1),
            dtype=self.linear_config.ssm_state_dtype,
            shape=self.linear_config.get_ssm_state_shape(),
            layer_num=self.linear_config.linear_layer_num,
            device="cuda",
        )
        return

    def init_linear_att_state(self, req: "InferReq"):
        conv_index = req.req_idx
        ssm_start = req.req_idx * (self.mtp_step + 1)
        self.req_to_conv_state.buffer[:, conv_index, ...].fill_(0)
        # #17: zero the FULL (mtp_step + 1)-row SSM block, not just canonical row +0, so a future
        # first-step verify reading offset>0 after fresh init never hits a never-written row (NaN).
        self.req_to_ssm_state.buffer[:, ssm_start : ssm_start + (self.mtp_step + 1), ...].fill_(0)
        if self.req_to_mtp_state_index is not None:
            self.req_to_mtp_state_index[req.req_idx] = 0
        return

    def get_mamba_cache(self, layer_idx_in_all: int):
        assert (
            0 <= layer_idx_in_all < self.linear_config.all_layer_num
        ), f"invalid transformer layer index {layer_idx_in_all}"
        layer_idx_in_linear = layer_idx_in_all - (layer_idx_in_all // self.linear_config.full_attention_interval)
        conv_states = self.req_to_conv_state.buffer[layer_idx_in_linear]
        ssm_states = self.req_to_ssm_state.buffer[layer_idx_in_linear]
        return conv_states, ssm_states

    def copy_big_page_buffer_to_linear_att_state(self, big_page_buffer_idx: int, req: "InferReq"):
        big_page_buffers: LinearAttCacheManager = self.mem_manager.linear_att_big_page_buffers

        conv_state, ssm_state = big_page_buffers.get_state_cache(buffer_idx=big_page_buffer_idx)
        conv_dest = req.req_idx
        ssm_dest = req.req_idx * (self.mtp_step + 1)
        conv_cache_width = conv_state.shape[-1]
        self.req_to_conv_state.buffer[:, conv_dest, ..., :conv_cache_width] = conv_state
        self.req_to_ssm_state.buffer[:, ssm_dest, ...] = ssm_state
        if self.req_to_mtp_state_index is not None:
            self.req_to_mtp_state_index[req.req_idx] = 0
        return

    def copy_small_page_buffer_to_linear_att_state(
        self, req: "InferReq", linear_att_small_page_buffers: LinearAttCacheManager
    ):
        conv_state, ssm_state = linear_att_small_page_buffers.get_state_cache(
            buffer_idx=req.shared_kv_node.small_page_buffer_idx
        )
        conv_dest = req.req_idx
        ssm_dest = req.req_idx * (self.mtp_step + 1)
        conv_cache_width = conv_state.shape[-1]
        # TODO 下面这个从 cpu cache 拷贝数据的 gpu的操作，是否是阻塞的操作。
        # 同时，非连续对象的拷贝，可能存在效率问题。
        self.req_to_conv_state.buffer[:, conv_dest, ..., :conv_cache_width] = conv_state
        self.req_to_ssm_state.buffer[:, ssm_dest, ...] = ssm_state
        if self.req_to_mtp_state_index is not None:
            self.req_to_mtp_state_index[req.req_idx] = 0
        return
