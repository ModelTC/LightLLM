import torch
from typing import List, Tuple
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.server.router.model_infer.mode_backend.generic_pre_process import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)


logger = init_logger(__name__)


class ChunkedPrefillBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.forward_step = 0
        args = get_env_start_args()
        self.max_wait_step = args.router_max_wait_tokens
        self.need_prefill_count = 0

    def prefill(self, reqs: List[Tuple]):
        self._init_reqs(reqs, init_req_obj=False)
        self.need_prefill_count += 1
        return

    def decode(self):
        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids
        )

        if aborted_reqs:
            g_infer_context.filter_reqs(aborted_reqs)

        # 先 decode
        if decode_reqs:
            kwargs, run_reqs = prepare_decode_inputs(decode_reqs)
            logits = self.model.forward(**kwargs)
            self._overlap_req_init_and_filter(
                uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True
            )
            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            self._post_handle(
                run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=False, do_filter_finished_reqs=False
            )
            logits = None

        # 再 prefill
        if len(decode_reqs) == 0 or (self.forward_step % self.max_wait_step == 0) or (self.need_prefill_count > 0):
            if prefill_reqs:
                self.need_prefill_count -= 1
                kwargs, run_reqs = prepare_prefill_inputs(
                    prefill_reqs, is_chuncked_mode=True, is_multimodal=self.is_multimodal
                )
                logits = self.model.forward(**kwargs)
                self._overlap_req_init_and_filter(
                    uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True
                )
                next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
                next_token_ids = next_token_ids.detach().cpu().numpy()
                next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
                self._post_handle(
                    run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=True, do_filter_finished_reqs=False
                )
                logits = None

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        self.forward_step += 1
        return
