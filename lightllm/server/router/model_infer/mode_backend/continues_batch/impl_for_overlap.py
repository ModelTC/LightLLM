import torch
from typing import List, Tuple
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import InferReq, InferSamplingParams, g_infer_context
from lightllm.server.core.objs import FinishStatus
from lightllm.utils.log_utils import init_logger
from .overlap_pre_process import overlap_prepare_inputs, overlap_prepare_prefill_inputs, overlap_prepare_decode_inputs
from .post_process import sample

logger = init_logger(__name__)

# 这个模式的gpu使用率很高，但是对首字会有一次decode时间的延迟。
class OverlapBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

    def prefill(self, reqs: List[Tuple]):
        self._init_reqs(reqs, init_req_obj=False)
        return

    def decode(self):
        uninit_reqs, finished_reqs, prefill_run_reqs, decode_run_reqs = overlap_prepare_inputs()

        if len(prefill_run_reqs) != 0:
            kwargs = overlap_prepare_prefill_inputs(prefill_run_reqs)
            logits = self.model.forward(**kwargs)
            next_token_ids, next_token_probs = sample(logits, prefill_run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            self.post_handle(prefill_run_reqs, next_token_ids, next_token_logprobs)

        if len(decode_run_reqs) != 0:
            kwargs = overlap_prepare_decode_inputs(decode_run_reqs)
            logits = self.model.forward(**kwargs)

            # 利用推理的时间，延迟折叠下一个请求的初始化和退出操作
            with torch.cuda.stream(g_infer_context.get_overlap_stream()):
                g_infer_context.filter([req.shm_req.request_id for req in finished_reqs])
                for req in uninit_reqs:
                    req.init_all()
            torch.cuda.current_stream().wait_stream(g_infer_context.get_overlap_stream())

            next_token_ids, next_token_probs = sample(logits, decode_run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            self.post_handle(decode_run_reqs, next_token_ids, next_token_logprobs)

        else:
            # 利用推理的时间，延迟折叠下一个请求的初始化和退出操作
            with torch.cuda.stream(g_infer_context.get_overlap_stream()):
                g_infer_context.filter([req.shm_req.request_id for req in finished_reqs])
                for req in uninit_reqs:
                    req.init_all()
            torch.cuda.current_stream().wait_stream(g_infer_context.get_overlap_stream())

        return

    def post_handle(self, run_reqs: List[InferReq], next_token_ids, next_token_logprobs):
        for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
            # prefill and decode is same
            req_obj: InferReq = req_obj
            req_obj.cur_kv_len = req_obj.get_cur_total_len()

            req_obj.set_next_gen_token_id(next_token_id, next_token_logprob)
            req_obj.cur_output_len += 1

            req_obj.out_token_id_count[next_token_id] += 1
            req_obj.update_finish_status(self.eos_id)

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
        return
