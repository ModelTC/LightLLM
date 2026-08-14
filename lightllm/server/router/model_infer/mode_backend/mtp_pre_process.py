import torch
from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.basemodel.triton_kernel.gen_mtp_prefill_params import gen_mtp_new_input_ids


def prepare_mtp_prefill_inputs(
    model_input: ModelInput,
    b_next_token_ids: torch.Tensor,
    mtp_draft_input_hiddens: torch.Tensor,
) -> ModelInput:
    # MTP supplies explicit token ids; mixed-prefill gathering must not replace them.
    model_input.b_is_decode_req = None
    model_input.input_ids = gen_mtp_new_input_ids(
        input_ids=model_input.input_ids,
        b_next_token_ids=b_next_token_ids,
        b_seq_len=model_input.b_seq_len,
        b_ready_cache_len=model_input.b_ready_cache_len,
    )
    model_input.mtp_draft_input_hiddens = mtp_draft_input_hiddens
    return model_input
