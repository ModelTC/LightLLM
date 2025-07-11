import torch
from typing import List, Tuple, Optional, Callable
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.mode_backend.continues_batch.impl import ContinuesBatchBackend
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventPack
from lightllm.server.router.model_infer.mode_backend.pre import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample

logger = init_logger(__name__)


class ChunkedPrefillBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

    def normal_prefill_reqs(
        self,
        event_pack: OverlapEventPack,
        prefill_reqs: List[InferReq],
    ):
        # 第一阶段
        model_input, run_reqs = prepare_prefill_inputs(
            prefill_reqs, is_chuncked_mode=not self.disable_chunked_prefill, is_multimodal=self.is_multimodal
        )
        model_output = self.model.forward(model_input)
        logits = model_output.logits

        if self.prefill_mask_func is not None:
            self.prefill_mask_func(run_reqs, logits)

        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_logprobs = torch.log(next_token_probs)
        sync_event = torch.cuda.Event()
        sync_event.record()

        # 第二阶段
        event_pack.notify_post_handle_and_wait_pre_post_handle()
        update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=not self.disable_chunked_prefill)

        # 第三阶段
        event_pack.notify_forward_and_wait_post_handle()
        sync_event.synchronize()
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = next_token_logprobs.detach().cpu().numpy()
        self._post_handle(
            run_reqs=run_reqs,
            next_token_ids=next_token_ids,
            next_token_logprobs=next_token_logprobs,
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
        )
        # 第四阶段
        event_pack.notify_pre_post_handle()
        return

    def normal_decode(
        self,
        event_pack: OverlapEventPack,
        decode_reqs: List[InferReq],
    ):
        model_input, run_reqs = prepare_decode_inputs(decode_reqs)
        model_output = self.model.forward(model_input)
        logits = model_output.logits

        if self.decode_mask_func is not None:
            self.decode_mask_func(run_reqs, logits)

        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_logprobs = torch.log(next_token_probs)
        sync_event = torch.cuda.Event()
        sync_event.record()

        # 第二阶段
        event_pack.notify_post_handle_and_wait_pre_post_handle()
        update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=False)

        # 第三阶段
        event_pack.notify_forward_and_wait_post_handle()
        sync_event.synchronize()
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = next_token_logprobs.detach().cpu().numpy()
        self._post_handle(
            run_reqs=run_reqs,
            next_token_ids=next_token_ids,
            next_token_logprobs=next_token_logprobs,
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
        )

        # 第四阶段
        event_pack.notify_pre_post_handle()
        return

    def infer_loop(self):
        torch.cuda.set_device(get_current_device_id())
        try:
            while True:
                event_pack = self.overlap_event_manager.get_overlap_event_pack()
                event_pack.wait_to_forward()

                self._try_read_new_reqs()

                prefill_reqs, decode_reqs = self._get_classed_reqs()
                if prefill_reqs:
                    self.normal_prefill_reqs(
                        event_pack=event_pack,
                        prefill_reqs=prefill_reqs,
                    )
                    continue

                if decode_reqs:
                    self.normal_decode(
                        event_pack=event_pack,
                        decode_reqs=decode_reqs,
                    )
                    continue

                event_pack.notify_post_handle_and_wait_pre_post_handle()
                event_pack.notify_forward_and_wait_post_handle()
                event_pack.notify_pre_post_handle()
                continue
        except BaseException as e:
            self.logger.exception(str(e))
