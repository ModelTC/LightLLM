from typing import TYPE_CHECKING, List

import torch

from lightllm.common.state_cache_manager import LayerCache, LinearAttCacheConfig, LinearAttCacheManager
from lightllm.utils.envs_utils import get_env_start_args

from .hybrid_base import HybridAttentionReqManager

from lightllm.backends import get_device_runtime, get_operator_backend

_device_runtime = get_device_runtime()
_operator_backend = get_operator_backend()


if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq


class ReqManagerForMamba(HybridAttentionReqManager):
    def __init__(self, max_request_num, max_sequence_length, mem_manager, linear_config: LinearAttCacheConfig):
        super().__init__(max_request_num, max_sequence_length, mem_manager)
        self.mtp_step = get_env_start_args().mtp_step
        # 因为在mtp的推理中，需要标记每个请求对应的mtp index状态(conv state 和 ssm state)，在mtp对应序列中
        # 的真实位置，所以需要需要一个标记来记录，不然算子无法找到真实的处理起点。
        self.req_to_mtp_state_index = (
            _device_runtime.zeros((max_request_num + 1,), dtype=torch.int32, device=_device_runtime.device)
            if self.mtp_step > 0
            else None
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
            device=_device_runtime.device,
        )
        self.req_to_ssm_state = LayerCache(
            size=(max_request_num + 1) * (self.mtp_step + 1),
            dtype=self.linear_config.ssm_state_dtype,
            shape=self.linear_config.get_ssm_state_shape(),
            layer_num=self.linear_config.linear_layer_num,
            device=_device_runtime.device,
        )
        return

    def init_hybrid_attention_state(self, req: "InferReq"):
        conv_index = req.req_idx
        ssm_start = req.req_idx * (self.mtp_step + 1)
        self.req_to_conv_state.buffer[:, conv_index, ...].fill_(0)
        # #17: zero the FULL (mtp_step + 1)-row SSM block, not just canonical row +0, so a future
        # first-step verify reading offset>0 after fresh init never hits a never-written row (NaN).
        self.req_to_ssm_state.buffer[:, ssm_start : ssm_start + (self.mtp_step + 1), ...].fill_(0)
        if self.req_to_mtp_state_index is not None:
            self.req_to_mtp_state_index[req.req_idx] = 0
        return

    def create_small_page_cache_manager(self, size: int):
        self.small_page_buffers = LinearAttCacheManager(size=size, linear_config=self.linear_config)
        return self.small_page_buffers

    def save_big_page_states(self, b_req_idx: torch.Tensor, req_indexes: List[int], buffer_indexes: List[int]):
        copy_linear_att_state_to_kv_buffer = _operator_backend.resolve(
            "common.basemodel.linear_att_copy.copy_linear_att_state_to_kv_buffer"
        )

        buffer_indexes = _device_runtime.move(
            _device_runtime.tensor(buffer_indexes, dtype=torch.int32, device="cpu"), non_blocking=True
        )
        state_cache_manager = self.big_page_buffers
        copy_linear_att_state_to_kv_buffer(
            b_req_idx=b_req_idx,
            big_page_buffer_ids=buffer_indexes,
            gpu_conv_state=self.req_to_conv_state.buffer,
            gpu_ssm_state=self.req_to_ssm_state.buffer,
            cpu_kv_conv_state=state_cache_manager.conv_state_cache.buffer,
            cpu_kv_ssm_state=state_cache_manager.ssm_state_cache.buffer,
            mtp_step=self.mtp_step,
        )
        return

    def save_state(self, req_idx: int, buffer_idx: int, state_cache_manager: LinearAttCacheManager):
        # checkpoint 只保存标准 conv 窗口和请求的基准 SSM 状态，不包含 MTP 扩展运行态。
        conv_cache_width = self.linear_config.get_conv_state_shape()[-1]
        gpu_conv_state = self.req_to_conv_state.buffer[:, req_idx, ..., :conv_cache_width]
        gpu_ssm_state = self.req_to_ssm_state.buffer[:, req_idx * (self.mtp_step + 1), ...]
        dst_conv_state, dst_ssm_state = state_cache_manager.get_state_cache(buffer_idx=buffer_idx)
        # Metal host checkpoints must be complete before the request state
        # advances. Only runtimes with pinned transfers use the async path.
        dst_conv_state.copy_(gpu_conv_state, non_blocking=_device_runtime.supports_pinned_memory)
        dst_ssm_state.copy_(gpu_ssm_state, non_blocking=_device_runtime.supports_pinned_memory)

    def get_mamba_cache(self, layer_idx_in_all: int):
        assert (
            0 <= layer_idx_in_all < self.linear_config.all_layer_num
        ), f"invalid transformer layer index {layer_idx_in_all}"
        layer_idx_in_linear = layer_idx_in_all - (layer_idx_in_all // self.linear_config.full_attention_interval)
        conv_states = self.req_to_conv_state.buffer[layer_idx_in_linear]
        ssm_states = self.req_to_ssm_state.buffer[layer_idx_in_linear]
        return conv_states, ssm_states

    def update_mtp_state(self, b_req_mtp_start_loc, b_req_idx, b_mtp_index, accepted_index, verify_width):
        linear_att_mtp_state_index_update = _operator_backend.resolve(
            "common.basemodel.mtp_utils.linear_att_mtp_state_index_update"
        )

        linear_att_mtp_state_index_update(
            req_to_mtp_state_index=self.req_to_mtp_state_index,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            b_req_idx=b_req_idx,
            b_mtp_index=b_mtp_index,
            accepted_index=accepted_index,
            verify_width=verify_width,
        )

    def restore_state(self, req: "InferReq", state_cache_manager: LinearAttCacheManager, buffer_idx: int):
        conv_state, ssm_state = state_cache_manager.get_state_cache(buffer_idx=buffer_idx)
        conv_dest = req.req_idx
        ssm_dest = req.req_idx * (self.mtp_step + 1)
        conv_cache_width = conv_state.shape[-1]
        self.req_to_conv_state.buffer[:, conv_dest, ..., :conv_cache_width] = conv_state
        self.req_to_ssm_state.buffer[:, ssm_dest, ...] = ssm_state
        if self.req_to_mtp_state_index is not None:
            self.req_to_mtp_state_index[req.req_idx] = 0
        return
