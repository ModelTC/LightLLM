import os
import shutil
import torch
from .impl import ContinuesBatchBackend
from typing import List, Tuple
from lightllm.server.core.objs import FinishStatus
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq, InferSamplingParams
from .pre_process import prepare_prefill_inputs, prepare_decode_inputs
from .post_process import sample
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class FirstTokenConstraintBackend(ContinuesBatchBackend):
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
        return

    def prefill(self, reqs: List[Tuple]):
        req_ids = self._init_reqs(reqs, init_req_obj=True)
        kwargs, run_reqs = prepare_prefill_inputs(req_ids, self.is_multimodal)
        logits = self.model.forward(**kwargs)
        # first token constraint
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[:, self.first_allowed_tokens] = False
        logits[mask] = -1000000.0

        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        self.post_handel(run_reqs, next_token_ids, next_token_logprobs)
        return

    def decode(self):
        kwargs, run_reqs = prepare_decode_inputs(g_infer_context.infer_req_ids)
        logits = self.model.forward(**kwargs)

        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        self.post_handel(run_reqs, next_token_ids, next_token_logprobs)
        return

    def post_handel(self, run_reqs: List[InferReq], next_token_ids, next_token_logprobs):
        finished_req_ids = []

        for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
            # prefill and decode is same
            req_obj: InferReq = req_obj
            req_obj.cur_kv_len = req_obj.get_cur_total_len()

            req_obj.set_next_gen_token_id(next_token_id, next_token_logprob)
            req_obj.cur_output_len += 1

            req_obj.out_token_id_count[next_token_id] += 1
            req_obj.update_finish_status(self.eos_id)

            if req_obj.finish_status.is_finished() or req_obj.shm_req.router_aborted:
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
