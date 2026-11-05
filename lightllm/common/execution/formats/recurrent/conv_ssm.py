from lightllm.common.execution.contracts.state import FormatSpec


def recurrent_format(dtype, ssm_dtype, shard, candidate_slots=1):
    import torch

    speculative = candidate_slots > 1
    return FormatSpec(
        "conv_ssm",
        dtype,
        "request_conv_ssm",
        ("conv", "ssm", "accepted_index") if speculative else ("conv", "ssm"),
        1,
        shard,
        field_dtypes=(dtype, ssm_dtype) + ((torch.int32,) if speculative else ()),
        candidate_slots=candidate_slots,
    )
