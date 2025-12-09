from numpy import ndarray


import torch
from .impl import ChunkedPrefillBackend
from typing import Any, List
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
        g_infer_context.use_hybrid_radix_cache = True

    @override
    def init_model(self, kvargs):
        from lightllm.server.router.dynamic_prompt.hybrid_radix_cache import HybridRadixCache

        super().init_model(kvargs)
        assert isinstance(self.radix_cache, HybridRadixCache)
        return

    def prefill_normal(
        self,
        event_pack: OverlapEventPack,
        prefill_reqs: List[InferReq],
    ):
        # 第一阶段: 模型推理
        model_input, run_reqs = prepare_prefill_inputs(
            prefill_reqs, is_chuncked_mode=not self.disable_chunked_prefill, is_multimodal=self.is_multimodal
        )
        with torch.cuda.stream(g_infer_context.get_overlap_stream()):
            model_output = self.model.forward(model_input)
            _, next_token_ids_cpu, next_token_logprobs_cpu = self._sample_and_scatter_token(
                logits=model_output.logits,
                b_req_idx=model_input.b_req_idx,
                b_mtp_index=model_input.b_mtp_index,
                run_reqs=run_reqs,
                is_prefill=True,
                b_prefill_has_output_cpu=model_input.b_prefill_has_output_cpu,
                mask_func=self.prefill_mask_func,
            )
            sync_event = torch.cuda.Event()
            sync_event.record()

        # 第二阶段
        event_pack.notify_post_handle_and_wait_pre_post_handle()
        update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=not self.disable_chunked_prefill)

        # 第三阶段
        event_pack.notify_forward_and_wait_post_handle()
        sync_event.synchronize()
        self._post_handle(
            run_reqs=run_reqs,
            next_token_ids=next_token_ids_cpu,
            next_token_logprobs=next_token_logprobs_cpu,
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
            nixl_prefill_chuncked_handle_func=self.nixl_prefill_chuncked_handle_func,
        )

        if not self.disable_chunked_prefill:
            for req in run_reqs:
                # NOTE 忽略完整的prefill, 因为请求文本全是system prompt 的情况应该比较小
                if req.cur_kv_len < req.get_cur_total_len() - 1:
                    self._handle_radix_cache_insert(req)

        # 第四阶段
        event_pack.notify_pre_post_handle()
        return

    def _handle_radix_cache_insert(self, req: "InferReq"):
        from lightllm.models.qwen3next.mem_manager import HaveStateBuffer
        from lightllm.models.qwen3next.req_manager import Qwen3NextReqManager

        assert isinstance(self.model.req_manager.mem_manager, HaveStateBuffer)
        assert isinstance(self.model.req_manager, Qwen3NextReqManager)

        # 获取当前 chunked_prefill 处理的 token IDs
        input_token_ids: Any | ndarray[Any, Any] = req.get_input_token_ids()
        key = torch.tensor(input_token_ids[0 : req.cur_kv_len], dtype=torch.int64, device="cpu")

        # 获取对应的 token 索引
        value = self.model.req_manager.req_to_token_indexs[req.req_idx][: req.cur_kv_len].cpu()

        buffer_idx = self.model.req_manager.req_to_buffer_indexes[req.req_idx].cpu()

        # 确保有足够的空间用于新的 buffer
        release_buffers = self.radix_cache.free_radix_cache_to_get_enough_token(0, 1)

        # 分配新的 buffer 并复制当前 buffer 的内容
        self.model.req_manager.mem_manager.free_state_cache_buffer(release_buffers)
        new_buffer_idx = self.model.req_manager.mem_manager.alloc_state_cache_buffer(1)[0]
        self.model.req_manager.mem_manager.copy_state_cache_buffer(buffer_idx, new_buffer_idx)
        self.model.req_manager.req_to_buffer_indexes[req.req_idx] = new_buffer_idx

        _, new_shared_kv_node = self.radix_cache.insert(key, value, buffer_idx)

        self.radix_cache.dec_node_ref_counter(req.shared_kv_node)
        self.radix_cache.add_node_ref_counter(new_shared_kv_node)
        req.shared_kv_node = new_shared_kv_node
