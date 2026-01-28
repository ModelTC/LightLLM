import torch
from typing import List, Tuple
from .impl import ChunkedPrefillBackend
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.mode_backend.pre import prepare_prefill_inputs
from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventPack


class RewardModelBackend(ChunkedPrefillBackend):
    def __init__(self) -> None:
        super().__init__()

        self.prefill = self.reward_prefill
        return

    def reward_prefill(self, event_pack: OverlapEventPack, prefill_reqs: List[InferReq]):

        assert self.disable_chunked_prefill is True
        model_input, run_reqs = prepare_prefill_inputs(prefill_reqs, is_chuncked_mode=not self.disable_chunked_prefill)

        model_output = self.model.forward(model_input)
        scores: torch.Tensor = model_output.logits
        scores = scores.unsqueeze(1).detach().cpu().float().numpy()

        next_token_id = 1
        next_token_logprob = 1.0

        for req_obj, score in zip(run_reqs, scores):
            # prefill and decode is same
            req_obj: InferReq = req_obj
            req_obj.cur_kv_len = req_obj.get_cur_total_len()

            req_obj.cur_output_len += 1
            req_obj.set_next_gen_token_id(next_token_id, next_token_logprob, output_len=req_obj.cur_output_len)
            req_obj.update_finish_status(self.eos_id, output_len=req_obj.cur_output_len)

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

        event_pack.notify_post_handle_and_wait_pre_post_handle()
        event_pack.notify_forward_and_wait_post_handle()
        event_pack.notify_pre_post_handle()
        return
