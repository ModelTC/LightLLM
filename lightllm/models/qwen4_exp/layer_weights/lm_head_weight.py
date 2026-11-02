from typing import Optional

import torch

from lightllm.common.basemodel.layer_weights.meta_weights import LMHeadWeight


class Qwen4ExpLMHeadWeight(LMHeadWeight):
    def _cuda_forward(
        self, input: torch.Tensor, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        rows = input.shape[1]
        if rows <= 1 or rows % 8 == 0 or input.dtype not in (torch.float16, torch.bfloat16):
            return super()._cuda_forward(input, out=out, alloc_func=alloc_func)

        tokens = input.transpose(0, 1)
        padded = torch.cat((tokens, tokens[:1].expand((-rows) % 8, -1)), dim=0)
        scores = super()._cuda_forward(padded.transpose(0, 1), alloc_func=alloc_func)
        shape = (self.weight.shape[0], rows)
        if out is None:
            out = alloc_func(shape, dtype=input.dtype, device=input.device)
        elif out.shape != shape or out.dtype != input.dtype or out.device != input.device:
            raise ValueError("LM-head output must match the unpadded shape, dtype and device")
        out.copy_(scores[:, :rows])
        return out
