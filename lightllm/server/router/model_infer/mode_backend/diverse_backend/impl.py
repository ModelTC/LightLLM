import torch
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.infer_batch import (
    g_infer_context,
    InferReq,
    InferReqGroup,
    InferSamplingParams,
)
from typing import List, Tuple
from lightllm.utils.log_utils import init_logger
from lightllm.server.tokenizer import get_tokenizer
from lightllm.server.req_id_generator import convert_sub_id_to_group_id
from lightllm.server.router.model_infer.mode_backend.generic_pre_process import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample


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
        group_reqs = [
            g_infer_context.requests_mapping[req_id]
            for req_id in req_ids
            if convert_sub_id_to_group_id(req_id) == req_id
        ]
        groups = [
            g_infer_context.group_mapping[req_id] for req_id in req_ids if convert_sub_id_to_group_id(req_id) == req_id
        ]
        kwargs, group_run_reqs = prepare_prefill_inputs(
            group_reqs, is_chuncked_mode=False, is_multimodal=self.is_multimodal
        )
        logits = self.model.forward(**kwargs)
        batch_idx, run_reqs = self.diverse_copy(groups)
        logits = logits[batch_idx]
        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        self._post_handle(
            run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=False, do_filter_finished_reqs=False
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
            logits = self.model.forward(**kwargs)

            self._overlap_req_init_and_filter(uninit_reqs=[], ok_finished_reqs=ok_finished_reqs, clear_list=True)

            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

            self._post_handle(
                run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=False, do_filter_finished_reqs=False
            )

        self._overlap_req_init_and_filter(uninit_reqs=[], ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return
