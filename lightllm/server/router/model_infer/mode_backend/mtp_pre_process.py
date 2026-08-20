import torch
from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.basemodel.triton_kernel.gen_mtp_prefill_params import gen_mtp_new_input_ids


def prepare_mtp_prefill_inputs(
    model_input: ModelInput,
    b_next_token_ids: torch.Tensor,
    mtp_draft_input_hiddens: torch.Tensor,
) -> ModelInput:
    # MTP supplies explicit token ids; mark every row as prefill so mixed-prefill
    # gathering does not replace them with request-level decode token ids.
    model_input.b_is_decode_req = torch.zeros_like(model_input.b_req_idx, dtype=torch.bool)
    model_input.input_ids = gen_mtp_new_input_ids(
        input_ids=model_input.input_ids,
        b_next_token_ids=b_next_token_ids,
        b_seq_len=model_input.b_seq_len,
        b_ready_cache_len=model_input.b_ready_cache_len,
    )
    model_input.mtp_draft_input_hiddens = mtp_draft_input_hiddens
    return model_input
