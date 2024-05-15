import torch
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import InferBatch, InferReq, InferSamplingParams
from .pre_process import splitfuse_prepare_decode_inputs
from lightllm.server.router.model_infer.mode_backend.continues_batch.post_process import sample


class SplitFuseBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

    @calculate_time(show=False, min_cost_ms=300)
    def prefill_batch(self, batch_id):
        """
        splitfuse 模式下prefill 没有实际操作。
        """
        return {}

    @calculate_time(show=True, min_cost_ms=200)
    def decode_batch(self, batch_id):
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        kwargs, decode_reqs, prefill_reqs = splitfuse_prepare_decode_inputs(
            batch, self.splitfuse_block_size, self.radix_cache
        )
        decode_req_num = len(decode_reqs)
        all_reqs = decode_reqs
        all_reqs.extend(prefill_reqs)

        logits = self.model.splitfuse_forward(**kwargs)
        next_token_ids, next_token_probs = sample(logits, all_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        index = 0
        for req_obj, next_token_id, next_token_logprob in zip(all_reqs, next_token_ids, next_token_logprobs):
            if index < decode_req_num:
                req_obj.cur_kv_len = len(req_obj.input_token_ids)
                req_obj.input_token_ids.append(next_token_id)
                req_obj.out_token_id_count[next_token_id] += 1
                req_obj.update_finish_status(self.eos_id)
                metadata = {
                    "id": int(next_token_id),
                    "logprob": float(next_token_logprob),
                }
                output_dict[req_obj.r_id] = (
                    req_obj.req_status,
                    req_obj.cur_kv_len,
                    req_obj.get_output_len(),
                    [(int(next_token_id), metadata)],
                    req_obj.finish_status.value,
                    None,
                )
            else:
                old_input_token_size = len(req_obj.input_token_ids)
                split_len = min(old_input_token_size - req_obj.cur_kv_len, self.splitfuse_block_size)
                if req_obj.cur_kv_len + split_len == old_input_token_size:
                    # 有输出
                    req_obj.cur_kv_len = old_input_token_size
                    req_obj.input_token_ids.append(next_token_id)
                    req_obj.out_token_id_count[next_token_id] += 1
                    req_obj.update_finish_status(self.eos_id)
                    metadata = {
                        "id": int(next_token_id),
                        "logprob": float(next_token_logprob),
                    }
                    output_dict[req_obj.r_id] = (
                        req_obj.req_status,
                        req_obj.cur_kv_len,
                        req_obj.get_output_len(),
                        [(int(next_token_id), metadata)],
                        req_obj.finish_status.value,
                        None,
                    )
                elif req_obj.cur_kv_len + split_len < old_input_token_size:
                    # 没输出
                    req_obj.cur_kv_len = req_obj.cur_kv_len + split_len
                    req_obj.update_finish_status(self.eos_id)
                    output_dict[req_obj.r_id] = (
                        req_obj.req_status,
                        req_obj.cur_kv_len,
                        req_obj.get_output_len(),
                        [],
                        req_obj.finish_status.value,
                        None,
                    )
                else:
                    assert False, "error state"
            index += 1

        self.cache[batch.batch_id] = batch
        return output_dict
