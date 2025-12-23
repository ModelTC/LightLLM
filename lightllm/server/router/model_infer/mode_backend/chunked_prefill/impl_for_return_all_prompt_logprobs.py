import torch
from .impl import ChunkedPrefillBackend
from typing import List
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.mode_backend.pre import prepare_prefill_inputs
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventPack


class ReturnPromptLogProbBackend(ChunkedPrefillBackend):
    def __init__(self) -> None:
        super().__init__()
        self.prefill = self.return_all_prompt_logprobs_prefill
        return

    def return_all_prompt_logprobs_prefill(self, event_pack: OverlapEventPack, prefill_reqs: List[InferReq]):

        # 在 return all_prompt_logprobs 的模式下，不能启用 dynamic prompt cache
        assert self.radix_cache is None
        assert self.disable_chunked_prefill is True

        model_input, run_reqs = prepare_prefill_inputs(prefill_reqs, is_chuncked_mode=not self.disable_chunked_prefill)

        model_output = self.model.forward(model_input)
        prompt_all_logits = model_output.logits

        input_ids = model_input.input_ids
        b_ready_cache_len = model_input.b_ready_cache_len
        b_seq_len = model_input.b_seq_len
        last_index = torch.cumsum(b_seq_len, dim=0, dtype=torch.long) - 1
        logits = prompt_all_logits[last_index, :]

        b_q_seq_len = b_seq_len - b_ready_cache_len
        b_start_loc = torch.cumsum(b_q_seq_len, dim=0, dtype=torch.long) - b_q_seq_len
        b_start_loc = b_start_loc.cpu().numpy()
        b_q_seq_len = b_q_seq_len.cpu().numpy()

        for req_obj, start_loc, q_seq_len in zip(run_reqs, b_start_loc, b_q_seq_len):
            req_obj: InferReq = req_obj
            cur_ids: torch.Tensor = input_ids[start_loc : start_loc + q_seq_len]
            cur_logits = prompt_all_logits[start_loc : start_loc + q_seq_len]
            cur_logprobs = torch.log_softmax(cur_logits, dim=-1, dtype=torch.float)[0:-1, :]
            cur_logprobs = torch.gather(cur_logprobs, dim=1, index=cur_ids[1:].view(-1, 1)).detach().cpu().numpy()

            if req_obj.shm_req.input_len > 1:
                if self.is_master_in_dp:
                    req_obj.shm_req.shm_logprobs.arr[1 : req_obj.shm_req.input_len] = cur_logprobs.flatten()

        if self.prefill_mask_func is not None:
            self.prefill_mask_func(run_reqs, logits)

        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        update_packs = self._pre_post_handle(run_reqs, is_chuncked_mode=not self.disable_chunked_prefill)
        self._post_handle(
            run_reqs=run_reqs,
            next_token_ids=next_token_ids,
            next_token_logprobs=next_token_logprobs,
            run_reqs_update_packs=update_packs,
            extra_post_req_handle_func=self.extra_post_req_handle_func,
        )

        event_pack.notify_post_handle_and_wait_pre_post_handle()
        event_pack.notify_forward_and_wait_post_handle()
        event_pack.notify_pre_post_handle()
        return
