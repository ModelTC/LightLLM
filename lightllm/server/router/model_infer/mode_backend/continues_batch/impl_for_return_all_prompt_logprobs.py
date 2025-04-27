import torch
from .impl import ContinuesBatchBackend
from typing import List, Tuple
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import InferReq, InferSamplingParams, g_infer_context
from lightllm.server.router.model_infer.mode_backend.generic_pre_process import prepare_prefill_inputs
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample


class ReturnPromptLogProbBackend(ContinuesBatchBackend):
    def __init__(self) -> None:
        super().__init__()

    def prefill(self, run_reqs: List[Tuple]):
        # 在 return all_prompt_logprobs 的模式下，不能启用 dynamic prompt cache
        assert self.radix_cache is None
        req_ids = self._init_reqs(run_reqs, init_req_obj=True)

        req_objs = self._trans_req_ids_to_req_objs(req_ids)
        kwargs, run_reqs = prepare_prefill_inputs(req_objs, is_chuncked_mode=False, is_multimodal=self.is_multimodal)

        prompt_all_logits = self.model.forward(**kwargs)
        input_ids = kwargs["input_ids"]
        b_ready_cache_len = kwargs["b_ready_cache_len"]
        b_seq_len = kwargs["b_seq_len"]
        last_index = torch.cumsum(b_seq_len, dim=0, dtype=torch.long) - 1
        logits = prompt_all_logits[last_index, :]

        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        b_q_seq_len = b_seq_len - b_ready_cache_len
        b_start_loc = torch.cumsum(b_q_seq_len, dim=0, dtype=torch.long) - b_q_seq_len
        b_start_loc = b_start_loc.cpu().numpy()
        b_q_seq_len = b_q_seq_len.cpu().numpy()

        finished_req_ids = []
        for req_obj, next_token_id, next_token_logprob, start_loc, q_seq_len in zip(
            run_reqs, next_token_ids, next_token_logprobs, b_start_loc, b_q_seq_len
        ):
            # prefill and decode is same
            req_obj: InferReq = req_obj
            req_obj.cur_kv_len = req_obj.get_cur_total_len()

            req_obj.set_next_gen_token_id(next_token_id, next_token_logprob)
            req_obj.cur_output_len += 1

            # 填充 logprobs 信息
            cur_ids: torch.Tensor = input_ids[start_loc : start_loc + q_seq_len]
            cur_logits = prompt_all_logits[start_loc : start_loc + q_seq_len]
            cur_logprobs = torch.log_softmax(cur_logits, dim=-1, dtype=torch.float)[0:-1, :]
            cur_logprobs = torch.gather(cur_logprobs, dim=1, index=cur_ids[1:].view(-1, 1)).detach().cpu().numpy()

            for i in range(req_obj.shm_req.input_len - 1):
                req_obj.shm_req.shm_logprobs.arr[i + 1] = cur_logprobs[i]

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

        g_infer_context.filter(finished_request_ids=finished_req_ids)
        return
