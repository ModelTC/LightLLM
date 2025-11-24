import torch
from .impl import ChunkedPrefillBackend
from typing import List
from typing_extensions import override
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventPack
from lightllm.server.router.model_infer.mode_backend.pre import (
    prepare_prefill_inputs,
)
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class HybridRadixCacheBackend(ChunkedPrefillBackend):
    def __init__(self) -> None:
        super().__init__()
        logger.info("Using HybridRadixCacheBackend for hybrid attention model.")
        self.extra_post_req_handle_func = self._handle_hybrid_radix_cache_insert

    @override
    def init_model(self, kvargs):
        from lightllm.server.router.dynamic_prompt.hybrid_radix_cache import HybridRadixCache
        super().init_model(kvargs)
        assert isinstance(self.radix_cache, HybridRadixCache)
        return

    def _handle_hybrid_radix_cache_insert(self, req_obj: "InferReq", next_token_id, next_token_logprob):
        # TODO : add docs
        if (req_obj.is_multi_chat_req or
            req_obj.cur_kv_len >= req_obj.get_cur_total_len()):
            return

        g_infer_state_lock.acquire()
        input_token_ids = req_obj.get_input_token_ids()
        key = torch.tensor(input_token_ids[0 : req_obj.cur_kv_len], dtype=torch.int64, device="cpu")

        value = self.model.req_manager.req_to_token_indexs[req_obj.req_idx][: req_obj.cur_kv_len].cpu()

        buffer_idx = self.model.req_manager.req_to_buffer_indexes[req_obj.req_idx].cpu()

        self.radix_cache.free_radix_cache_to_get_enough_token(0, 1)

        new_buffer_idx = self.model.req_manager.mem_manager.alloc_state_cache_buffer(1)[0]
        self.model.req_manager.mem_manager.copy_state_cache_buffer(buffer_idx, new_buffer_idx)
        self.model.req_manager.req_to_buffer_indexes[req_obj.req_idx] = new_buffer_idx

        _, new_shared_kv_node = self.radix_cache.insert(key, value, buffer_idx)

        self.radix_cache.dec_node_ref_counter(req_obj.shared_kv_node)
        self.radix_cache.add_node_ref_counter(new_shared_kv_node)
        req_obj.shared_kv_node = new_shared_kv_node
        g_infer_state_lock.release()
