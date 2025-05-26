import os
import shutil
import torch
import functools

from .impl import ChunkedPrefillBackend
from lightllm.server.core.objs import FinishStatus
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.server.router.model_infer.mode_backend.generic_pre_process import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.server.tokenizer import get_tokenizer
from typing import List, Tuple
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


class OutlinesConstraintBackend(ChunkedPrefillBackend):
    def __init__(self) -> None:
        super().__init__()

    def init_custom(self):
        # remove outlines cache
        if self.rank_in_node == 0:
            cache_path = os.path.join(os.path.expanduser("~"), ".cache/outlines")
            if os.path.exists(cache_path) and os.path.isdir(cache_path):
                shutil.rmtree(cache_path)
                logger.info("outlines cache dir is removed")
            else:
                logger.info("outlines cache dir is not exist")

        from outlines.models.transformers import TransformerTokenizer

        self.tokenizer = TransformerTokenizer(
            get_tokenizer(self.args.model_dir, self.args.tokenizer_mode, trust_remote_code=self.args.trust_remote_code)
        )
        eos_token_ids = []
        eos_token_ids.append(self.tokenizer.eos_token_id)
        eos_token_ids.extend(self.args.eos_id)
        # 附加一个 eos_token_ids 数组，然后利用 .outlines_patch 中的实现，修改一outlines的默认实现
        # 添加多eos_id 的逻辑
        self.tokenizer.eos_token_ids = eos_token_ids
        logger.info(f"eos_ids {self.tokenizer.eos_token_ids}")

        @functools.lru_cache(maxsize=200)
        def get_cached_regex_guide(regex: str):
            from outlines.fsm.guide import RegexGuide

            logger.info(f"regex_guide cache miss for '{regex}'")
            return RegexGuide.from_regex(regex, self.tokenizer)

        self.get_cached_regex_guide = get_cached_regex_guide
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

            self._init_guide_infos(run_reqs)
            all_has_no_constraint = all([not e.sampling_param.has_constraint_setting() for e in run_reqs])
            if not all_has_no_constraint:
                mask = torch.ones_like(logits, dtype=torch.bool)
                for i, run_obj in enumerate(run_reqs):
                    self._mask_req_out_token(i, run_obj, mask)
                logits[mask] = -1000000.0

            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            self._post_handle(
                run_reqs,
                next_token_ids,
                next_token_logprobs,
                is_chuncked_mode=False,
                do_filter_finished_reqs=False,
                extra_post_req_handle_func=self._update_state_fsm,
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
                # 对于不能满足前缀匹配的logic位置，将其logics设置为一个较大负值，将其概率掩盖为 0
                self._init_guide_infos(run_reqs)
                mask = torch.ones_like(logits, dtype=torch.bool)
                for i, run_obj in enumerate(run_reqs):
                    self._mask_req_out_token(i, run_obj, mask)

                logits[mask] = -1000000.0

                next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
                next_token_ids = next_token_ids.detach().cpu().numpy()
                next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
                self._post_handle(
                    run_reqs,
                    next_token_ids,
                    next_token_logprobs,
                    is_chuncked_mode=True,
                    do_filter_finished_reqs=False,
                    extra_post_req_handle_func=self._update_state_fsm,
                )
                logits = None

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        self.forward_step += 1
        return

    def _update_state_fsm(self, req_obj: InferReq, next_token_id, next_token_logprob):
        next_token_id = int(next_token_id)
        if req_obj.sampling_param.regular_constraint is not None:
            sample_params = req_obj.sampling_param
            regex_guide = sample_params.regex_guide
            sample_params.fsm_current_state = regex_guide.get_next_state(sample_params.fsm_current_state, next_token_id)
            if sample_params.fsm_current_state == -1:
                req_obj.finish_status.set_status(FinishStatus.FINISHED_STOP)
        return

    def _mask_req_out_token(self, i, run_obj: InferReq, mask):
        from outlines.fsm.guide import RegexGuide

        if run_obj.get_chuncked_input_token_len() == run_obj.get_cur_total_len():
            # this run_obj is ready to gen next token.
            sample_params = run_obj.sampling_param
            if sample_params.regular_constraint is not None:
                regex_guide: RegexGuide = sample_params.regex_guide
                ok_token_id_list = regex_guide.get_next_instruction(sample_params.fsm_current_state).tokens
                mask[i, ok_token_id_list] = False
            elif sample_params.allowed_token_ids is not None:
                mask[i, sample_params.allowed_token_ids] = False
            else:
                mask[i, :] = False
        else:
            # no constraint
            mask[i, :] = False
        return

    def _init_guide_infos(self, run_reqs: List[InferReq]):
        for i, run_obj in enumerate(run_reqs):
            run_obj: InferReq = run_obj
            sample_params = run_obj.sampling_param
            if sample_params.regular_constraint is not None:
                if not hasattr(sample_params, "regex_guide"):
                    sample_params.regex_guide = self.get_cached_regex_guide(sample_params.regular_constraint)
