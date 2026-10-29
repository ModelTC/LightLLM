from copy import copy

from lightllm.common.state_cache_manager import LayerCache
from lightllm.utils.envs_utils import get_env_start_args

from .linear_att import ReqManagerForMamba
from .linear_att_pd_state import LinearAttPDState


class Glm5NextReqManager(ReqManagerForMamba):
    """KDA runtime state and the NSA indexer's incomplete four-token pool."""

    kda = True

    ##### 管理器初始化与检查点缓存池 #####

    def __init__(self, max_request_num, max_sequence_length, mem_manager, linear_config):
        # Both checkpoint sizes are multiples of this hash page. A restored
        # prefix therefore has no incomplete K-pool to serialize or replay.
        assert (
            get_env_start_args().linear_att_hash_page_size % linear_config.index_kpool == 0
        ), "GLM K-pool requires cache pages aligned to index_kpool"
        args = get_env_start_args()
        state_args = None
        if args.run_mode == "prefill" and args.ssm_state_mode == "native":
            # This role never verifies speculative tokens. Retain draft model
            # KV, but only one committed KDA state and the base conv window.
            state_args = copy(args)
            state_args.mtp_step = 0
        super().__init__(max_request_num, max_sequence_length, mem_manager, linear_config, state_args=state_args)
        tail_step = self.mtp_step
        if args.run_mode in ("prefill", "decode"):
            # PD transports the raw ring as a fixed layout. P and D can use
            # different verify widths; modulo addressing must still agree.
            # KDA validation supports at most 15 speculative tokens.
            if not 0 <= tail_step <= 15:
                raise ValueError("GLM PD indexer state supports mtp_step between 0 and 15")
            tail_step = 15
        self.req_to_indexer_tail = LayerCache(
            size=max_request_num + 1,
            # Raw keys/gates are computation state, even when packed KV is FP8.
            dtype=linear_config.conv_state_dtype,
            shape=(linear_config.index_kpool + tail_step, 2 * linear_config.index_head_dim),
            layer_num=linear_config.get_full_att_kv_layer_num_with_draft_model(),
            device="cuda",
        )
        self.pd_state = LinearAttPDState(self.linear_state_manager, self.req_to_indexer_tail)

    ##### 请求运行态初始化 #####

    def init_hybrid_attention_state(self, req):
        super().init_hybrid_attention_state(req)
        self.req_to_indexer_tail.buffer[:, req.req_idx].zero_()

    ##### 检查点恢复：CPU 缓存 → 请求 GPU 运行态 #####

    def restore_state(self, req, state_cache_manager, buffer_idx, checkpoint_len=None):
        super().restore_state(req, state_cache_manager, buffer_idx, checkpoint_len)
        self.req_to_indexer_tail.buffer[:, req.req_idx].zero_()

    ##### Indexer 尾部运行态访问 #####

    def get_indexer_tail_buffer(self, layer_index):
        return self.req_to_indexer_tail.buffer[self.linear_config.get_full_att_kv_layer_index(layer_index)]
