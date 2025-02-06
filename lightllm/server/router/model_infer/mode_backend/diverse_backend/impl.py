import torch
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.infer_batch import (
    g_infer_context,
    InferReq,
    InferReqGroup,
    InferSamplingParams,
)
from typing import List, Tuple
from lightllm.server.core.objs import FinishStatus
from lightllm.utils.log_utils import init_logger
from lightllm.server.tokenizer import get_tokenizer
from lightllm.server.req_id_generator import convert_sub_id_to_group_id
from ..continues_batch.pre_process import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from ..continues_batch.post_process import sample


class DiversehBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

    def init_custom(self):
        pass

    def build_group(self, req_ids: List[int]):
        for r_id in req_ids:
            req: InferReq = g_infer_context.requests_mapping[r_id]
            group_req_id = req.shm_req.group_req_id
            if group_req_id not in g_infer_context.group_mapping:
                g_infer_context.group_mapping[group_req_id] = InferReqGroup(group_req_id=group_req_id)
            g_infer_context.group_mapping[group_req_id].add_req(r_id)

    def diverse_copy(self, groups: List[InferReqGroup]):
        batch_idx = []
        run_reqs = []
        for i in range(len(groups)):
            req_group = groups[i]
            best_of = req_group.best_of()
            if best_of > 1:
                req_group.diverse_copy(g_infer_context.req_manager, is_prefill=True)
                batch_idx.extend([i for _ in range(best_of)])
            else:
                batch_idx.append(i)
            run_reqs.extend(req_group.get_all_reqs())
        return batch_idx, run_reqs

    def prefill(self, reqs: List[Tuple]):
        req_ids = self._init_reqs(reqs)
        self.build_group(req_ids)
        group_req_ids = [req_id for req_id in req_ids if convert_sub_id_to_group_id(req_id) == req_id]
        groups = [
            g_infer_context.group_mapping[req_id] for req_id in req_ids if convert_sub_id_to_group_id(req_id) == req_id
        ]
        kwargs, group_run_reqs = prepare_prefill_inputs(group_req_ids, self.is_multimodal)
        logits = self.model.forward(**kwargs)
        batch_idx, run_reqs = self.diverse_copy(groups)
        logits = logits[batch_idx]
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
            req_obj: InferReq = req_obj
            req_obj.cur_kv_len = req_obj.get_cur_total_len()

            req_obj.set_next_gen_token_id(next_token_id, next_token_logprob)
            req_obj.cur_output_len += 1

            req_obj.out_token_id_count[next_token_id] += 1
            req_obj.update_finish_status(self.eos_id)

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
