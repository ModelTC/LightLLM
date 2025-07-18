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
from lightllm.common.basemodel.triton_kernel.gather_token_id import scatter_token
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager
from ..chunked_prefill.impl import ChunkedPrefillBackend


class DiversehBackend(ChunkedPrefillBackend):
    def __init__(self) -> None:
        super().__init__()
        self.prefill = self.beam_prefill
        self.classed_req_strict_prefill = True

    def diverse_copy(self, groups: List[InferReqGroup]) -> Tuple[List[int], List[InferReq]]:
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
        # 第一阶段
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

        with torch.cuda.stream(g_infer_context.get_overlap_stream()):

            model_output = self.model.forward(model_input)
            logits = model_output.logits

            batch_idx, run_reqs = self.diverse_copy(groups)
            b_req_idx = [req.req_idx for req in run_reqs]
            b_has_out = [model_input.b_prefill_has_output_cpu[i] for i in batch_idx]

            batch_idx = g_pin_mem_manager.gen_from_list(key="batch_idx_", data=batch_idx, dtype=torch.int64).cuda(
                non_blocking=True
            )
            b_req_idx = g_pin_mem_manager.gen_from_list(key="b_req_idx_", data=b_req_idx, dtype=torch.int32).cuda(
                non_blocking=True
            )
            b_has_out = g_pin_mem_manager.gen_from_list(key="b_has_out_", data=b_has_out, dtype=torch.bool).cuda(
                non_blocking=True
            )

            logits = logits[batch_idx]
            b_mtp_index = model_input.b_mtp_index[batch_idx]

            next_token_ids, next_token_logprobs = sample(logits, run_reqs, self.eos_id)

            scatter_token(
                next_token_ids=next_token_ids,
                req_to_next_token_ids=self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
                b_req_idx=b_req_idx,
                b_mtp_index=b_mtp_index,
                b_has_out=b_has_out,
            )

            next_token_ids_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                key="next_token_ids",
                gpu_tensor=next_token_ids,
            )
            next_token_logprobs_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
                key="next_token_logprobs",
                gpu_tensor=next_token_logprobs,
            )
            sync_event = torch.cuda.Event()
            sync_event.record()

        # 第二阶段
        event_pack.notify_post_handle_and_wait_pre_post_handle()
        update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=not self.disable_chunked_prefill)

        # 第三阶段
        event_pack.notify_forward_and_wait_post_handle()
        sync_event.synchronize()
        self._post_handle(
            run_reqs=run_reqs,
            next_token_ids=next_token_ids_cpu,
            next_token_logprobs=next_token_logprobs_cpu,
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
        )
        # 第四阶段
        event_pack.notify_pre_post_handle()
        return
