from lightllm.platform.ops import SamplingOps

from .common import unsupported_op


ASCEND_SAMPLING_OPS = SamplingOps(
    apply_penalty=unsupported_op("sampling.apply_penalty"),
    apply_penalty_gpu_cache=unsupported_op("sampling.apply_penalty_gpu_cache"),
    apply_invalid_token_ids=unsupported_op("sampling.apply_invalid_token_ids"),
    token_id_counter=unsupported_op("sampling.token_id_counter"),
    update_req_to_token_id_counter=unsupported_op("sampling.update_req_to_token_id_counter"),
)
