import torch
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.infer_batch import (
    g_infer_context,
    InferReq,
    InferReqGroup,
)
from typing import List, Tuple
from lightllm.server.req_id_generator import convert_sub_id_to_group_id
from lightllm.server.router.model_infer.mode_backend.pre import (
    prepare_prefill_inputs,
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventPack


class DiversehBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.prefill = self.beam_prefill

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

    def beam_prefill(self, event_pack: OverlapEventPack, prefill_reqs: List[InferReq]):
        group_reqs = [
            g_infer_context.requests_mapping[req.req_id]
            for req in prefill_reqs
            if convert_sub_id_to_group_id(req.req_id) == req.req_id
        ]
        groups = [
            g_infer_context.group_mapping[req.req_id]
            for req in prefill_reqs
            if convert_sub_id_to_group_id(req.req_id) == req.req_id
        ]
        model_input, group_run_reqs = prepare_prefill_inputs(
            group_reqs, is_chuncked_mode=not self.disable_chunked_prefill, is_multimodal=self.is_multimodal
        )
        model_output = self.model.forward(model_input)
        logits = model_output.logits

        batch_idx, run_reqs = self.diverse_copy(groups)
        logits = logits[batch_idx]

        next_token_ids_gpu, next_token_probs_gpu = sample(model_output.logits, run_reqs, self.eos_id)
        next_token_ids_cpu = next_token_ids_gpu.detach().cpu().numpy()
        next_token_logprobs_cpu = torch.log(next_token_probs_gpu).detach().cpu().numpy()

        update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=not self.disable_chunked_prefill)
        self._post_handle(
            run_reqs=run_reqs,
            next_token_ids=next_token_ids_cpu,
            next_token_logprobs=next_token_logprobs_cpu,
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
        )
        return
