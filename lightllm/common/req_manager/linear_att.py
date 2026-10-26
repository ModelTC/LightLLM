from typing import TYPE_CHECKING, List, Optional

import torch

from lightllm.common.linear_state import LinearStateManager
from lightllm.common.state_cache_manager import LinearAttCacheConfig, LinearAttCacheManager
from lightllm.utils.envs_utils import get_env_start_args

from .hybrid_base import HybridAttentionReqManager
from .linear_att_pd_state import LinearAttPDState


if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq


class ReqManagerForMamba(HybridAttentionReqManager):
    kda = False

    ##### 管理器初始化与检查点缓存池 #####

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
        self.linear_state_manager = LinearStateManager(max_request_num, linear_config, args, kda=self.kda)
        self.pd_state = LinearAttPDState(self.linear_state_manager)

    def create_small_page_cache_manager(self, size: int):
        self.small_page_buffers = LinearAttCacheManager(size=size, linear_config=self.linear_config)
        return self.small_page_buffers

    ##### 请求运行态初始化 #####

    def init_hybrid_attention_state(self, req: "InferReq"):
        self.linear_state_manager.reset(req.req_idx)

    ##### 检查点恢复：CPU 缓存 → 请求 GPU 运行态 #####

    def restore_state(self, req: "InferReq", state_cache_manager: LinearAttCacheManager, buffer_idx: int, checkpoint_len: Optional[int] = None):
        self.linear_state_manager.restore(req.req_idx, *state_cache_manager.get_state_cache(buffer_idx))

    ##### 检查点保存：请求 GPU 运行态 → CPU 缓存 #####

    def save_big_page_states(self, b_req_idx: torch.Tensor, req_indexes: List[int], buffer_indexes: List[int], checkpoint_lens: Optional[List[int]] = None):
        self.linear_state_manager.save_checkpoints(b_req_idx, req_indexes, buffer_indexes, self.big_page_buffers)

    def save_state(self, req_idx: int, buffer_idx: int, state_cache_manager: LinearAttCacheManager, checkpoint_len: Optional[int] = None):
        self.linear_state_manager.copy_accepted_state_to(req_idx, *state_cache_manager.get_state_cache(buffer_idx))

    ##### MTP 验证后的运行态更新 #####

    def update_mtp_state(
        self, b_req_mtp_start_loc, b_req_idx, b_mtp_index, accepted_index, verify_width, linear_state_batches=None
    ):
        self.linear_state_manager.commit_accepted_state(
            b_req_mtp_start_loc, b_req_idx, b_mtp_index, accepted_index, verify_width, linear_state_batches
        )

    ##### PD 分离：传输块与请求运行态的读写 #####

    def get_pd_state_size(self) -> int:
        return self.pd_state.state_nbytes

    def get_pd_state(self, req_idx: int, state_block: torch.Tensor, tp_rank: int):
        self.pd_state.get_pd_state(req_idx, state_block, tp_rank)

    def load_pd_state(self, req_idx: int, state_block: torch.Tensor, tp_rank: int):
        self.pd_state.load_pd_state(req_idx, state_block, tp_rank)

    ##### 计算层运行态访问 #####

    def get_mamba_cache(self, layer_idx_in_all: int):
        return self.linear_state_manager.get_layer_buffers(layer_idx_in_all)
