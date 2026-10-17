from typing import TYPE_CHECKING, List

import torch

from lightllm.common.recurrent_state import RecurrentState
from lightllm.common.state_cache_manager import LinearAttCacheConfig, LinearAttCacheManager
from lightllm.utils.envs_utils import get_env_start_args

from .hybrid_base import HybridAttentionReqManager


if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq


class ReqManagerForMamba(HybridAttentionReqManager):
    kda = False

    def __init__(self, max_request_num, max_sequence_length, mem_manager, linear_config: LinearAttCacheConfig):
        super().__init__(max_request_num, max_sequence_length, mem_manager)
        args = get_env_start_args()
        self.mtp_step = args.mtp_step
        # Prefill kernels read the canonical convolution window. Mixed prefill
        # and MTP decode still require a separate execution-layout change.
        if self.mtp_step > 0:
            assert args.enable_prefill_decode_mixed is False
        self.big_page_token_num = args.linear_att_page_block_num * args.linear_att_hash_page_size
        self.linear_config = linear_config
        self.recurrent_state = RecurrentState(max_request_num, linear_config, args, kda=self.kda)

    def init_hybrid_attention_state(self, req: "InferReq"):
        self.recurrent_state.reset(req.req_idx)

    def create_small_page_cache_manager(self, size: int):
        self.small_page_buffers = LinearAttCacheManager(size=size, linear_config=self.linear_config)
        return self.small_page_buffers

    def save_big_page_states(self, b_req_idx: torch.Tensor, req_indexes: List[int], buffer_indexes: List[int]):
        self.recurrent_state.save_checkpoints(b_req_idx, req_indexes, buffer_indexes, self.big_page_buffers)

    def save_state(self, req_idx: int, buffer_idx: int, state_cache_manager: LinearAttCacheManager):
        self.recurrent_state.export(req_idx, *state_cache_manager.get_state_cache(buffer_idx))

    def get_mamba_cache(self, layer_idx_in_all: int):
        return self.recurrent_state.layer_buffers(layer_idx_in_all)

    def update_mtp_state(
        self, b_req_mtp_start_loc, b_req_idx, b_mtp_index, accepted_index, verify_width, recurrent_state_batches=None
    ):
        self.recurrent_state.commit(
            b_req_mtp_start_loc, b_req_idx, b_mtp_index, accepted_index, verify_width, recurrent_state_batches
        )

    def restore_state(self, req: "InferReq", state_cache_manager: LinearAttCacheManager, buffer_idx: int):
        self.recurrent_state.restore(req.req_idx, *state_cache_manager.get_state_cache(buffer_idx))
