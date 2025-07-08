import torch
from .impl import ContinuesBatchBackend
from typing import List, Tuple, Callable, Optional
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import InferReq, InferSamplingParams, g_infer_context
from lightllm.server.router.model_infer.mode_backend.pre import prepare_prefill_inputs
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample


class ReturnPromptLogProbBackend(ContinuesBatchBackend):
    def __init__(self) -> None:
        super().__init__()

    def prefill(self, run_reqs: List[Tuple]):
        # 在 return all_prompt_logprobs 的模式下，不能启用 dynamic prompt cache
        assert self.radix_cache is None
        self._init_reqs(run_reqs, init_req_obj=False)
        return

    def normal_prefill_reqs(
        self,
        prefill_reqs: List[InferReq],
        uninit_reqs: List[InferReq],
        ok_finished_reqs: List[InferReq],
        mask_func: Optional[Callable[[List[InferReq], torch.Tensor], None]] = None,
        extra_post_req_handle_func: Optional[Callable[[InferReq, int, float], None]] = None,
    ):
        model_input, run_reqs = prepare_prefill_inputs(
            prefill_reqs, is_chuncked_mode=False, is_multimodal=self.is_multimodal
        )

        model_output = self.model.forward(model_input)
        prompt_all_logits = model_output.logits

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)

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
                req_obj.shm_req.shm_logprobs.arr[1 : req_obj.shm_req.input_len] = cur_logprobs.flatten()

        if mask_func is not None:
            mask_func(run_reqs, logits)

        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        self._post_handle(
            run_reqs,
            next_token_ids,
            next_token_logprobs,
            is_chuncked_mode=False,
            do_filter_finished_reqs=False,
            extra_post_req_handle_func=extra_post_req_handle_func,
        )
        return
