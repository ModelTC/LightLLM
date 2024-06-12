import torch


def layernorm_forward(x, weight, eps):
    return torch.layer_norm(x, (x.shape[-1],), weight, bias=None, eps=eps)


def multi_head_layernorm_forward(x, weight, eps):
    inp_dtype = x.dtype
    x = x.to(torch.float32)
    mean = x.mean(-1, keepdim=True)
    variance = (x - mean).pow(2).mean(-1, keepdim=True)
    x = (x - mean) * torch.rsqrt(variance + eps)
    x = weight.to(torch.float32) * x
    return x.to(inp_dtype)
