import torch
import copy
from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.basemodel.triton_kernel.gen_mtp_prefill_params import gen_mtp_new_input_ids


def prepare_mtp_prefill_inputs(
    model_input: ModelInput, b_next_token_ids: torch.Tensor, mtp_draft_input_hiddens: torch.Tensor
):
    new_model_input = copy.copy(model_input)
    new_input_ids = gen_mtp_new_input_ids(
        input_ids=model_input.input_ids,
        b_next_token_ids=b_next_token_ids,
        b_seq_len=model_input.b_seq_len,
        b_ready_cache_len=model_input.b_ready_cache_len,
    )
    new_model_input.input_ids = new_input_ids
    new_model_input.mtp_draft_input_hiddens = mtp_draft_input_hiddens
    return new_model_input
