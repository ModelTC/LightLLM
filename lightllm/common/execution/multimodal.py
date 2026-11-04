"""Bind embedding storage once and prepare image/audio indices before computation."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MultimodalInputs:
    cache: torch.Tensor
    device: torch.device

    @classmethod
    def bind(cls, text_weight, cache=None):
        if cache is None:
            cache = torch.empty((0, 0, text_weight.shape[1]), dtype=text_weight.dtype, device=text_weight.device)
        if cache.ndim != 3 or cache.shape[2] != text_weight.shape[1] or cache.dtype != text_weight.dtype:
            raise ValueError("multimodal embedding cache disagrees with text embedding dimensions or dtype")
        return cls(cache, text_weight.device)

    def prepare(self, state):
        state.cpu_embed_cache_tensor = self.cache
        if not state.is_prefill:
            return
        objects = {}
        for params in state.multimodal_params or ():
            for item in params["images"] + params["audios"]:
                objects.setdefault(item["token_id"], item)
        starts, lengths, offsets = [], [], []
        for token_id, item in objects.items():
            length, offset = item["token_num"], item["start_index_in_embed_cache"]
            if length <= 0 or offset < 0 or offset + length > self.cache.shape[0] or self.cache.shape[1] == 0:
                raise ValueError("multimodal input references data outside the bound embedding cache")
            starts.append(token_id)
            lengths.append(length)
            offsets.append(offset)
        for name, values in (
            ("img_start_token_ids", starts),
            ("img_token_lens", lengths),
            ("img_start_locs_in_cache", offsets),
        ):
            tensor = torch.tensor(values, dtype=torch.long, device="cpu", pin_memory=self.device.type == "cuda")
            setattr(state, name, tensor.to(self.device, non_blocking=True))
