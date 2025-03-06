import os
import shutil
import torch
from typing import List, Tuple

from .impl import ContinuesBatchBackend
from .pre_process import prepare_prefill_inputs, prepare_decode_inputs
from .post_process import sample

from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.core.objs import FinishStatus
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq, InferSamplingParams
from lightllm.server.tokenizer import get_tokenizer
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class XgrammarBackend(ContinuesBatchBackend):
    def __init__(self) -> None:
        super().__init__()

    def init_custom(self):
        import xgrammar as xgr

        self.tokenizer = get_tokenizer(
            self.args.model_dir, self.args.tokenizer_mode, trust_remote_code=self.args.trust_remote_code
        )

        tokenizer_info = xgr.TokenizerInfo.from_huggingface(self.tokenizer)
        self.xgrammar_compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)
        self.xgrammar_token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

        eos_token_ids = []
        eos_token_ids.append(self.tokenizer.eos_token_id)
        eos_token_ids.extend(self.args.eos_id)
        return

    @calculate_time(show=False, min_cost_ms=300)
    def prefill(self, reqs: List[Tuple]):
        import xgrammar as xgr

        req_ids = self._init_reqs(reqs)
        kwargs, run_reqs = prepare_prefill_inputs(req_ids, is_multimodal=self.is_multimodal)

        logics = self.model.forward(**kwargs)

        for i, run_obj in enumerate(run_reqs):
            run_obj: InferReq = run_obj
            sample_params = run_obj.sampling_param
            if sample_params.guided_grammar is not None:
                xgrammar_compiled_grammar = self.xgrammar_compiler.compile_grammar(sample_params.guided_grammar)
                sample_params.xgrammar_matcher = xgr.GrammarMatcher(xgrammar_compiled_grammar)
            elif sample_params.guided_json is not None:
                xgrammar_compiled_grammar = self.xgrammar_compiler.compile_json_schema(sample_params.guided_json)
                sample_params.xgrammar_matcher = xgr.GrammarMatcher(xgrammar_compiled_grammar)
            self._mask_req_out_token(i, run_obj, logics[i])

        # fix the logics with -inf to a large negative value
        logics[logics == float("-inf")] = -1000000.0

        next_token_ids, next_token_probs = sample(logics, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        self.post_handel(run_reqs, next_token_ids, next_token_logprobs)

        return

    @calculate_time(show=True, min_cost_ms=200)
    def decode(self):
        import xgrammar as xgr

        kwargs, run_reqs = prepare_decode_inputs(g_infer_context.infer_req_ids)
        run_reqs: List[InferReq] = run_reqs

        logits = self.model.forward(**kwargs)

        all_has_no_constraint = all([not e.sampling_param.has_constraint_setting() for e in run_reqs])
        if not all_has_no_constraint:
            for i, run_obj in enumerate(run_reqs):
                self._mask_req_out_token(i, run_obj, logits[i])

        logits[logits == float("-inf")] = -1000000.0
        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        self.post_handel(run_reqs, next_token_ids, next_token_logprobs)
        return

    def post_handel(self, run_reqs: List[InferReq], next_token_ids, next_token_logprobs):
        import xgrammar as xgr

        finished_req_ids = []

        for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
            # prefill and decode is same
            req_obj: InferReq = req_obj
            req_obj.cur_kv_len = req_obj.get_cur_total_len()

            req_obj.set_next_gen_token_id(next_token_id, next_token_logprob)
            req_obj.cur_output_len += 1

            req_obj.out_token_id_count[next_token_id] += 1
            req_obj.update_finish_status(self.eos_id)

            matcher = req_obj.sampling_param.xgrammar_matcher
            assert matcher.accept_token(next_token_id)

            if req_obj.finish_status.is_finished() or req_obj.shm_req.router_aborted or matcher.is_terminated():
                finished_req_ids.append(req_obj.shm_req.request_id)

            if self.is_master_in_dp:
                # shm_cur_kv_len shm_cur_output_len 是 router 调度进程需要读的信息
                # finish_token_index finish_status candetoken_out_len 是
                # detokenization 进程需要的信息，注意这些变量的写入顺序避免异步协同问题。
                req_obj.shm_req.shm_cur_kv_len = req_obj.cur_kv_len
                req_obj.shm_req.shm_cur_output_len = req_obj.cur_output_len

                if req_obj.finish_status.is_finished():
                    req_obj.shm_req.finish_token_index = req_obj.get_cur_total_len() - 1
                    req_obj.shm_req.finish_status = req_obj.finish_status

                req_obj.shm_req.candetoken_out_len = req_obj.cur_output_len

        g_infer_context.filter(finished_req_ids)
        return

    def _mask_req_out_token(self, i, run_obj: InferReq, logits):
        import xgrammar as xgr

        sample_params = run_obj.sampling_param
        if sample_params.guided_grammar is not None or sample_params.guided_json is not None:
            sample_params.xgrammar_matcher.fill_next_token_bitmask(self.xgrammar_token_bitmask)
            xgr.apply_token_bitmask_inplace(logits, self.xgrammar_token_bitmask.to(logits.device))
        return
