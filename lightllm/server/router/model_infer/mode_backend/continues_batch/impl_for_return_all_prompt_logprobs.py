import torch
from .impl import ContinuesBatchBackend
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import InferBatch, InferReq, InferSamplingParams
from .pre_process import prepare_prefill_inputs
from .post_process import sample


class ReturnPromptLogProbBackend(ContinuesBatchBackend):
    def __init__(self) -> None:
        super().__init__()

    @calculate_time(show=False, min_cost_ms=300)
    def prefill_batch(self, batch_id):
        # 在 return all_prompt_logprobs 的模式下，不能启用 dynamic prompt cache
        assert self.radix_cache is None
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        kwargs, run_reqs = prepare_prefill_inputs(batch, self.radix_cache, self.model.mem_manager)

        prompt_all_logits = self.model.forward(**kwargs)
        input_ids = kwargs["input_ids"]
        b_start_loc = kwargs["b_start_loc"]
        b_seq_len = kwargs["b_seq_len"]
        last_index = torch.cumsum(b_seq_len, dim=0, dtype=torch.long) - 1
        logits = prompt_all_logits[last_index, :]

        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        b_start_loc = b_start_loc.cpu().numpy()
        b_seq_len = b_seq_len.cpu().numpy()
        for req_obj, next_token_id, next_token_logprob, start_loc, seq_len in zip(
            run_reqs, next_token_ids, next_token_logprobs, b_start_loc, b_seq_len
        ):
            # prefill and decode is same
            req_obj.cur_kv_len = len(req_obj.input_token_ids)
            req_obj.input_token_ids.append(next_token_id)
            req_obj.out_token_id_count[next_token_id] += 1
            metadata = {
                "id": int(next_token_id),
                "logprob": float(next_token_logprob),
            }

            cur_ids: torch.Tensor = input_ids[start_loc : start_loc + seq_len]
            cur_logits = prompt_all_logits[start_loc : start_loc + seq_len]
            cur_logprobs = torch.log_softmax(cur_logits, dim=-1, dtype=torch.float)[0:-1, :]
            cur_logprobs = torch.gather(cur_logprobs, dim=1, index=cur_ids[1:].view(-1, 1)).detach().cpu().numpy()

            cur_ids = cur_ids.cpu().numpy()
            all_prompts = []
            for index in range(len(cur_ids) - 1):
                tmp_dict = {int(cur_ids[index + 1]): float(cur_logprobs[index, 0])}
                all_prompts.append([int(cur_ids[index]), tmp_dict])

            req_obj.update_finish_status(self.eos_id)

            metadata["prompt_logprobs"] = all_prompts
            metadata["prompt_token_ids"] = [int(e) for e in cur_ids]
            output_dict[req_obj.r_id] = (
                req_obj.req_status,
                req_obj.cur_kv_len,
                req_obj.get_output_len(),
                [(int(next_token_id), metadata)],
                req_obj.finish_status.value,
                None,
            )  # 请求状态， 当前占用的kv的长度， 当前输出token的数量， 输出的token的id和元信息列表， 是否推理结束的状态， 额外保留参数

        self.cache[batch.batch_id] = batch
        return output_dict
