from lightllm.common.basemodel.triton_kernel.gen_sampling_params import (
    token_id_counter,
    update_req_to_token_id_counter,
)
from lightllm.common.basemodel.triton_kernel.post_process.apply_invalid_token import apply_invalid_token_ids
from lightllm.common.basemodel.triton_kernel.post_process.apply_penalty import apply_penalty
from lightllm.common.basemodel.triton_kernel.post_process.apply_penalty_gpu_cache import apply_penalty_gpu_cache
from lightllm.platform.ops.sampling import SamplingOps


CUDA_LIKE_SAMPLING_OPS = SamplingOps(
    apply_penalty=apply_penalty,
    apply_penalty_gpu_cache=apply_penalty_gpu_cache,
    apply_invalid_token_ids=apply_invalid_token_ids,
    token_id_counter=token_id_counter,
    update_req_to_token_id_counter=update_req_to_token_id_counter,
)
