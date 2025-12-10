import torch
import numpy as np
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel.infer_struct import InferStateInfo


class Qwen3VLInferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self.input_ids = None
        self.deepstack_features = []
        self.deepstack_end_layer = None
        self.img_start_token_ids = []
        self.img_token_lens = []
        self.img_start_locs = []

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        InferStateInfo.init_some_extra_state(self, model, input_ids)
        pos = self.position_ids[None, :].expand(3, -1)
        cos_T = torch.index_select(model._cos_cached, 0, pos[0])  # [L, d/2]
        cos_H = torch.index_select(model._cos_cached, 0, pos[1])
        cos_W = torch.index_select(model._cos_cached, 0, pos[2])
        sin_T = torch.index_select(model._sin_cached, 0, pos[0])
        sin_H = torch.index_select(model._sin_cached, 0, pos[1])
        sin_W = torch.index_select(model._sin_cached, 0, pos[2])
        cos_half = self.apply_interleaved_mrope(
            torch.stack([cos_T, cos_H, cos_W], dim=0), model.mrope_section
        )  # [L, d/2]
        sin_half = self.apply_interleaved_mrope(
            torch.stack([sin_T, sin_H, sin_W], dim=0), model.mrope_section
        )  # [L, d/2]

        self.position_cos = torch.cat([cos_half, cos_half], dim=-1).contiguous()  # [L, d]
        self.position_sin = torch.cat([sin_half, sin_half], dim=-1).contiguous()
        if self.is_prefill:
            pos = None
        return
