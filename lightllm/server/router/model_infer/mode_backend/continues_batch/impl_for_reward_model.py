import torch
from typing import List, Tuple
from .impl import ContinuesBatchBackend
from lightllm.server.router.model_infer.infer_batch import InferReq, InferSamplingParams, g_infer_context
from lightllm.server.router.model_infer.mode_backend.pre import prepare_prefill_inputs
from lightllm.server.core.objs import FinishStatus


class RewardModelBackend(ContinuesBatchBackend):
    def __init__(self) -> None:
        super().__init__()

    def decode(self):
        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids
        )

        if aborted_reqs:
            g_infer_context.filter_reqs(aborted_reqs)

        if prefill_reqs:
            self._prefill_reqs(req_objs=prefill_reqs)

        if decode_reqs:
            self.normal_decode(decode_reqs=decode_reqs, uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs)

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return

    def _prefill_reqs(self, req_objs: List[InferReq]):
        model_input, run_reqs = prepare_prefill_inputs(
            req_objs, is_chuncked_mode=False, is_multimodal=self.is_multimodal
        )

        model_output = self.model.forward(model_input)
        scores: torch.Tensor = model_output.logits
        scores = scores.unsqueeze(1).detach().cpu().float().numpy()

        next_token_id = 1
        next_token_logprob = 1.0

        finished_req_ids = []

        for req_obj, score in zip(run_reqs, scores):
            # prefill and decode is same
            req_obj: InferReq = req_obj
            req_obj.cur_kv_len = req_obj.get_cur_total_len()

            req_obj.set_next_gen_token_id(next_token_id, next_token_logprob)
            req_obj.cur_output_len += 1

            req_obj.update_finish_status(self.eos_id)

            if req_obj.finish_status.is_finished() or req_obj.shm_req.router_aborted:
                finished_req_ids.append(req_obj.shm_req.request_id)

            if self.is_master_in_dp:
                # 写入 reward_score
                req_obj.shm_req.reward_score = score

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
