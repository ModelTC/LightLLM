import numpy as np

from .logprob_utils import logprob_info


def get_prompt_logprobs_dtype(topk: int):
    return np.dtype(
        [
            ("top_token_ids", np.int32, (topk,)),
            ("top_logprobs", np.float32, (topk,)),
        ]
    )


def build_prompt_logprobs_metadata(req, tokenizer=None):
    prompt_token_ids = req.shm_prompt_ids.arr[: req.input_len]
    if req.input_len <= 1:
        return {
            "prompt_logprobs": [None],
            "prompt_token_ids": [int(token_id) for token_id in prompt_token_ids],
        }

    topk = req.sample_params.prompt_logprobs
    prompt_logprobs = [None]
    if topk == 0:
        prompt_logprobs.extend({} for _ in range(req.input_len - 1))
        return {
            "prompt_logprobs": prompt_logprobs,
            "prompt_token_ids": [int(token_id) for token_id in prompt_token_ids],
        }

    captured = req.link_prompt_logprobs_shm_array()

    for row_index in range(req.input_len - 1):
        position_logprobs = {}
        for index in range(topk):
            top_token_id = int(captured["top_token_ids"][row_index, index])
            if top_token_id >= 0:
                position_logprobs[top_token_id] = logprob_info(
                    tokenizer,
                    top_token_id,
                    captured["top_logprobs"][row_index, index],
                    index + 1,
                )

        prompt_logprobs.append(position_logprobs)

    return {
        "prompt_logprobs": prompt_logprobs,
        "prompt_token_ids": [int(token_id) for token_id in prompt_token_ids],
    }
