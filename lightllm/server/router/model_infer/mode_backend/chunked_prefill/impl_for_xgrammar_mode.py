import functools
import torch
from typing import List
from .impl import ChunkedPrefillBackend
from lightllm.utils.infer_utils import calculate_time
from lightllm.server.core.objs import FinishStatus
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.server.tokenizer import get_tokenizer
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class XgrammarBackend(ChunkedPrefillBackend):
    def __init__(self) -> None:
        super().__init__()
        self.support_overlap = False
        self.prefill_mask_func = self._prefill_mask_callback
        self.decode_mask_func = self._decode_mask_callback
        self.extra_post_req_handle_func = self._update_xgrammer_fsm

    def init_custom(self):
        import xgrammar as xgr

        self.tokenizer = get_tokenizer(
            self.args.model_dir, self.args.tokenizer_mode, trust_remote_code=self.args.trust_remote_code
        )

        self.tokenizer_info = xgr.TokenizerInfo.from_huggingface(self.tokenizer)
        self.xgrammar_compiler = xgr.GrammarCompiler(self.tokenizer_info, max_threads=8)
        self.xgrammar_token_bitmask = xgr.allocate_token_bitmask(1, self.tokenizer_info.vocab_size)

        @functools.lru_cache(maxsize=200)
        def get_cached_grammar(type: str, grammar: str):
            logger.info(f"grammar cache miss for {type}: '{grammar}'")
            try:
                if type == "grammar":
                    if grammar == "json":
                        return self.xgrammar_compiler.compile_builtin_json_grammar()

                    else:
                        return self.xgrammar_compiler.compile_grammar(grammar)
                elif type == "schema":
                    return self.xgrammar_compiler.compile_json_schema(grammar)
                else:
                    raise ValueError(f"Unknown xgrammar type: {type}")
            except Exception as e:
                logger.error(f"Failed to compile {type}: {e}")
                raise

        self.get_cached_grammar = get_cached_grammar
        return

    def _decode_mask_callback(self, run_reqs: List[InferReq], logits: torch.Tensor):
        self._init_req_xgrammer_matcher_infos(run_reqs=run_reqs)
        all_has_no_constraint = all([not e.sampling_param.has_constraint_setting() for e in run_reqs])
        if not all_has_no_constraint:
            for i, run_obj in enumerate(run_reqs):
                self._mask_req_out_token(i, run_obj, logits[i])

        logits[logits == float("-inf")] = -1000000.0
        # mask out the padding token logits
        logits[:, self.tokenizer_info.vocab_size :] = -1000000.0
        return

    def _prefill_mask_callback(self, run_reqs: List[InferReq], logits: torch.Tensor):
        self._init_req_xgrammer_matcher_infos(run_reqs=run_reqs)
        for i, run_obj in enumerate(run_reqs):
            self._mask_req_out_token(i, run_obj, logits[i])

        # fix the logics with -inf to a large negative value
        logits[logits == float("-inf")] = -1000000.0
        # mask out the padding token logits
        logits[:, self.tokenizer_info.vocab_size :] = -1000000.0
        return

    def _update_xgrammer_fsm(self, req_obj: InferReq, next_token_id, next_token_logprob):
        import xgrammar as xgr

        if not hasattr(req_obj.sampling_param, "xgrammar_matcher"):
            return

        matcher = req_obj.sampling_param.xgrammar_matcher
        assert matcher.accept_token(next_token_id)
        if matcher.is_terminated():
            req_obj.finish_status.set_status(FinishStatus.FINISHED_STOP)
        return

    def _mask_req_out_token(self, i, run_obj: InferReq, logits: torch.Tensor):
        import xgrammar as xgr

        if run_obj.get_chuncked_input_token_len() == run_obj.get_cur_total_len():
            sample_params = run_obj.sampling_param
            if sample_params.guided_grammar is not None or sample_params.guided_json is not None:
                sample_params.xgrammar_matcher.fill_next_token_bitmask(self.xgrammar_token_bitmask)
                xgr.apply_token_bitmask_inplace(logits, self.xgrammar_token_bitmask.to(logits.device))
        return

    def _init_req_xgrammer_matcher_infos(self, run_reqs: List[InferReq]):
        import xgrammar as xgr

        for i, run_obj in enumerate(run_reqs):
            run_obj: InferReq = run_obj
            sample_params = run_obj.sampling_param
            if sample_params.guided_grammar is not None:
                if not hasattr(sample_params, "xgrammar_matcher"):
                    ctx = self.get_cached_grammar("grammar", sample_params.guided_grammar)
                    sample_params.xgrammar_matcher = xgr.GrammarMatcher(ctx)
            elif sample_params.guided_json is not None:
                if not hasattr(sample_params, "xgrammar_matcher"):
                    ctx = self.get_cached_grammar("schema", sample_params.guided_json)
                    sample_params.xgrammar_matcher = xgr.GrammarMatcher(ctx)
        return
