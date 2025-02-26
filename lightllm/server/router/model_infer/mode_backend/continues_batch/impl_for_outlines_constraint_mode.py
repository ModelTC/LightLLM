import os
import shutil
import torch
from .impl import ContinuesBatchBackend
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.core.objs import FinishStatus
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq, InferSamplingParams
from .pre_process import prepare_prefill_inputs, prepare_decode_inputs
from .post_process import sample
from lightllm.server.tokenizer import get_tokenizer
from typing import List, Tuple
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class OutlinesConstraintBackend(ContinuesBatchBackend):
    def __init__(self) -> None:
        super().__init__()

    def init_custom(self):
        # 导入修改 outlines 的部分默认实现
        import lightllm.server.router.model_infer.mode_backend.continues_batch.outlines_patch.impl as _nouse_

        # remove outlines cache
        if self.tp_rank == 0:
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

        kwargs, run_reqs = prepare_prefill_inputs(req_ids, is_multimodal=self.is_multimodal)
        run_reqs: List[InferReq] = run_reqs

        logics = self.model.forward(**kwargs)

        # 对于不能满足前缀匹配的logic位置，将其logics设置为一个较大负值，将其概率掩盖为 0
        mask = torch.ones_like(logics, dtype=torch.bool)
        for i, run_obj in enumerate(run_reqs):
            run_obj: InferReq = run_obj
            sample_params = run_obj.sampling_param
            if sample_params.regular_constraint is not None:
                sample_params.regex_guide = RegexGuide(sample_params.regular_constraint, self.tokenizer)
            self._mask_req_out_token(i, run_obj, mask)

        logics[mask] = -1000000.0

        next_token_ids, next_token_probs = sample(logics, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        self.post_handel(run_reqs, next_token_ids, next_token_logprobs)

        return

    def decode(self):
        kwargs, run_reqs = prepare_decode_inputs(g_infer_context.infer_req_ids)
        run_reqs: List[InferReq] = run_reqs

        logits = self.model.forward(**kwargs)

        all_has_no_constraint = all([not e.sampling_param.has_constraint_setting() for e in run_reqs])
        if not all_has_no_constraint:
            mask = torch.ones_like(logits, dtype=torch.bool)
            for i, run_obj in enumerate(run_reqs):
                self._mask_req_out_token(i, run_obj, mask)
            logits[mask] = -1000000.0

        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        self.post_handel(run_reqs, next_token_ids, next_token_logprobs)
        return

    def post_handel(self, run_reqs, next_token_ids, next_token_logprobs):
        finished_req_ids = []

        for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
            # prefill and decode is same
            req_obj: InferReq = req_obj
            req_obj.cur_kv_len = req_obj.get_cur_total_len()

            req_obj.set_next_gen_token_id(next_token_id, next_token_logprob)
            req_obj.cur_output_len += 1

            req_obj.out_token_id_count[next_token_id] += 1
            req_obj.update_finish_status(self.eos_id)

            self._update_state_fsm(req_obj, next_token_id)

            if req_obj.finish_status.is_finished() or req_obj.shm_req.router_aborted:
                finished_req_ids.append(req_obj.shm_req.request_id)

            if self.tp_rank < self.dp_size:
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

    def _update_state_fsm(self, req_obj: InferReq, next_token_id):
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
