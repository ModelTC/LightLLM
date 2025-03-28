import os
import shutil
import torch
from .impl import ContinuesBatchBackend
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.core.objs import FinishStatus
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq, InferSamplingParams
from lightllm.server.router.model_infer.mode_backend.generic_pre_process import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.server.tokenizer import get_tokenizer
from typing import List, Tuple
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class OutlinesConstraintBackend(ContinuesBatchBackend):
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
        return

    def prefill(self, reqs: List[Tuple]):

        req_ids = self._init_reqs(reqs)

        # import here, 当你不使用这个模式，缺少这些依赖也可以运行
        from outlines.fsm.guide import RegexGuide

        req_objs = self._trans_req_ids_to_req_objs(req_ids)
        kwargs, run_reqs = prepare_prefill_inputs(req_objs, is_chuncked_mode=False, is_multimodal=self.is_multimodal)
        run_reqs: List[InferReq] = run_reqs

        logics = self.model.forward(**kwargs)

        # 对于不能满足前缀匹配的logic位置，将其logics设置为一个较大负值，将其概率掩盖为 0
        mask = torch.ones_like(logics, dtype=torch.bool)
        for i, run_obj in enumerate(run_reqs):
            run_obj: InferReq = run_obj
            sample_params = run_obj.sampling_param
            if sample_params.regular_constraint is not None:
                sample_params.regex_guide = RegexGuide.from_regex(sample_params.regular_constraint, self.tokenizer)
            self._mask_req_out_token(i, run_obj, mask)

        logics[mask] = -1000000.0

        next_token_ids, next_token_probs = sample(logics, run_reqs, self.eos_id)
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

        return

    def decode(self):
        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids
        )
        assert len(uninit_reqs) == 0
        assert len(prefill_reqs) == 0

        if aborted_reqs:
            g_infer_context.filter_reqs(aborted_reqs)

        if decode_reqs:
            kwargs, run_reqs = prepare_decode_inputs(decode_reqs)
            run_reqs: List[InferReq] = run_reqs

            logits = self.model.forward(**kwargs)

            self._overlap_req_init_and_filter(uninit_reqs=[], ok_finished_reqs=ok_finished_reqs, clear_list=True)

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

        self._overlap_req_init_and_filter(uninit_reqs=[], ok_finished_reqs=ok_finished_reqs, clear_list=True)
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

        sample_params = run_obj.sampling_param
        if sample_params.regular_constraint is not None:
            regex_guide: RegexGuide = sample_params.regex_guide
            ok_token_id_list = regex_guide.get_next_instruction(sample_params.fsm_current_state).tokens
            mask[i, ok_token_id_list] = False
        elif sample_params.allowed_token_ids is not None:
            mask[i, sample_params.allowed_token_ids] = False
        else:
            mask[i, :] = False
        return
