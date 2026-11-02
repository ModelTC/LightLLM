import torch


def combine_mrope(cos: torch.Tensor, sin: torch.Tensor, sections):
    if cos.shape != sin.shape or cos.device != sin.device or cos.dtype != sin.dtype:
        raise ValueError("Rotary cosine and sine shapes, devices and dtypes must match")
    if cos.ndim == 2:
        return cos.repeat(1, 2).contiguous(), sin.repeat(1, 2).contiguous()
    if cos.ndim != 3 or cos.shape[0] != 3 or len(sections) != 3:
        raise ValueError("Expected three rotary axes and three mRoPE sections")
    offsets = torch.arange(cos.shape[-1], device=cos.device)
    height = ((offsets % 3) == 1) & (offsets <= 3 * sections[1])
    width = ((offsets % 3) == 2) & (offsets <= 3 * sections[2])
    combined = []
    for tensor in (cos, sin):
        values = torch.where(height, tensor[1], torch.where(width, tensor[2], tensor[0]))
        combined.append(values.repeat(1, 2).contiguous())
    return tuple(combined)
