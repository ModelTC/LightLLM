import torch
import copy
from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.basemodel.triton_kernel.gen_mtp_prefill_params import gen_mtp_new_input_ids


def prepare_mtp_prefill_inputs(
    model_input: ModelInput, b_next_token_ids: torch.Tensor, mtp_draft_input_hiddens: torch.Tensor
):
    # enable_prefill_decode_mixed 模式下，decode 请求混合在 prefill 请求中。
    # 但是mtp的input_ids已经是恢复ok，已经是正常的input_ids, 所以移除掉 b_is_decode_req。
    # 防止在 forward 阶段，因为 b_is_decode_req 不为空，导致 input_ids 被特殊处理。
    new_model_input = copy.copy(model_input)
    new_model_input.b_is_decode_req = None

    new_input_ids = gen_mtp_new_input_ids(
        input_ids=model_input.input_ids,
        b_next_token_ids=b_next_token_ids,
        b_seq_len=model_input.b_seq_len,
        b_ready_cache_len=model_input.b_ready_cache_len,
    )
    new_model_input.input_ids = new_input_ids
    new_model_input.mtp_draft_input_hiddens = mtp_draft_input_hiddens
    return new_model_input
