import torch


def get_torch_dtype(data_type: str) -> torch.dtype:
    if data_type in ["fp16", "float16"]:
        return torch.float16
    elif data_type in ["bf16", "bfloat16"]:
        return torch.bfloat16
    elif data_type in ["fp32", "float32"]:
        return torch.float32
    else:
        raise ValueError(f"Unsupport datatype {data_type}!")
