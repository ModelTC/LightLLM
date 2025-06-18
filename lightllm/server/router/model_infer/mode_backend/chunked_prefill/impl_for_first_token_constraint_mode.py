import os
import shutil
import torch
from .impl import ChunkedPrefillBackend
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.server.router.model_infer.mode_backend.pre import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class FirstTokenConstraintBackend(ChunkedPrefillBackend):
    def __init__(self) -> None:
        super().__init__()

    def init_custom(self):
        first_allowed_tokens_strs: str = os.environ.get("FIRST_ALLOWED_TOKENS", None)
        logger.info(f"first_allowed_tokens_strs : {first_allowed_tokens_strs}")
        # 使用该模式需要设置FIRST_ALLOWED_TOKENS 环境变量，格式为 "1,2" 或 "1,2,3"  等数字字符串
        assert first_allowed_tokens_strs is not None
        first_allowed_tokens_strs.split(",")
        self.first_allowed_tokens = [int(e.strip()) for e in first_allowed_tokens_strs.split(",") if len(e.strip()) > 0]
        logger.info(f"first_allowed_tokens : {self.first_allowed_tokens}")
        # check token_id < vocab_size
        assert all(e < self.model.vocab_size for e in self.first_allowed_tokens)
        self.fill_value = torch.tensor(-1000000.0)
        return

    def decode(self):
        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids
        )

        if aborted_reqs:
            g_infer_context.filter_reqs(aborted_reqs)

        # 先 decode
        if decode_reqs:
            model_input, run_reqs = prepare_decode_inputs(decode_reqs)
            model_output = self.model.forward(model_input)
            logits = model_output.logits
            self._overlap_req_init_and_filter(
                uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True
            )
            self._mask_first_gen_token_logits(run_reqs, logits)
            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            self._post_handle(
                run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=False, do_filter_finished_reqs=False
            )
            del model_output
            del logits

        # 再 prefill
        if len(decode_reqs) == 0 or (self.forward_step % self.max_wait_step == 0) or (self.need_prefill_count > 0):
            if prefill_reqs:
                self.need_prefill_count -= 1
                model_input, run_reqs = prepare_prefill_inputs(
                    prefill_reqs, is_chuncked_mode=True, is_multimodal=self.is_multimodal
                )
                model_output = self.model.forward(model_input)
                logits = model_output.logits
                self._overlap_req_init_and_filter(
                    uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True
                )
                self._mask_first_gen_token_logits(run_reqs, logits)
                next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
                next_token_ids = next_token_ids.detach().cpu().numpy()
                next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
                self._post_handle(
                    run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=True, do_filter_finished_reqs=False
                )
                del model_output
                del logits

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        self.forward_step += 1
        return

    def _mask_first_gen_token_logits(self, run_reqs: List[InferReq], logits: torch.Tensor):
        # 这个函数中的实现会造成全局同步，造成折叠失效，主要是
        # mask[i, self.first_allowed_tokens] 切片复制还有
        # 后续出现 .cpu() .cuda() 等操作也会造成异常的全局同步
        # to do remove all zeros_like, ones_like, use triton kernel
        # replace
        if any([req.cur_output_len == 0 for req in run_reqs]):
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for i, req in enumerate(run_reqs):
                if req.cur_output_len == 0:
                    mask[i, :] = True
                    mask[i, self.first_allowed_tokens] = False
            # 不能使用 logits[mask] = -1000000.0
            # 会存在诡异的多流异步问题, 可能是torch的bug
            new_logits = torch.where(mask, self.fill_value, logits)
            logits.copy_(new_logits)
        return
