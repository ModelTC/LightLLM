from typing import List

import torch

from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class NeoChatInferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None
        self.position_cos_h = None
        self.position_sin_h = None
        self.position_cos_w = None
        self.position_sin_w = None

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        if self.is_prefill:
            self.b_image_token_end = torch.zeros(
                self.position_ids.size(0),
                dtype=torch.int32,
                device=self.position_ids.device,
            )
            self.position_ids = self.get_neo_position(self.multimodal_params)
            self.b_image_token_tag = self.b_image_token_end > 0
        else:
            b_position_delta = [0 for _ in range(self.b_seq_len.shape[0])]
            for batch_idx, params in enumerate(self.multimodal_params or []):
                b_position_delta[batch_idx] = sum(
                    image["grid_thwd"][3] for image in params.get("images", [])
                )
            position_delta = torch.tensor(
                b_position_delta,
                dtype=self.position_ids.dtype,
                device=self.position_ids.device,
            )
            position_ids = self.position_ids + position_delta
            self.position_ids = position_ids.unsqueeze(0).expand(3, -1).clone()
            self.position_ids[1:].zero_()

        self.position_ids = self.position_ids.contiguous()
        self.position_cos = torch.index_select(model._cos_cached, 0, self.position_ids[0])
        self.position_sin = torch.index_select(model._sin_cached, 0, self.position_ids[0])
        self.position_cos_h = torch.index_select(model._hw_cos_cached, 0, self.position_ids[1])
        self.position_sin_h = torch.index_select(model._hw_sin_cached, 0, self.position_ids[1])
        self.position_cos_w = torch.index_select(model._hw_cos_cached, 0, self.position_ids[2])
        self.position_sin_w = torch.index_select(model._hw_sin_cached, 0, self.position_ids[2])
        return

    def get_neo_position(self, multimodal_params: List[dict]) -> torch.Tensor:
        """Reference Torch implementation of Neo's packed T/H/W positions.

        This intentionally favors correctness over launch efficiency.  It mirrors
        the former Triton kernel while remaining valid on CUDA and Ascend.
        """
        position_ids = self.position_ids.new_zeros((3, self.position_ids.size(0)))
        position_ids[0].copy_(self.position_ids)
        batch_size = int(self.b_q_seq_len.shape[0])
        params = list(multimodal_params or [])
        if len(params) < batch_size:
            params.extend({"images": [], "audios": []} for _ in range(batch_size - len(params)))
        if len(params) != batch_size:
            raise ValueError(f"multimodal batch size mismatch: expected {batch_size}, got {len(params)}")

        ready_cache_lens = self.b_ready_cache_len.detach().to(device="cpu").tolist()
        q_seq_lens = self.b_q_seq_len.detach().to(device="cpu").tolist()
        q_start_locs = self.b_q_start_loc.detach().to(device="cpu").tolist()

        for batch_idx, request_params in enumerate(params):
            images = request_params.get("images", [])
            cache_len = int(ready_cache_lens[batch_idx])
            q_seq_len = int(q_seq_lens[batch_idx])
            packed_start = int(q_start_locs[batch_idx])
            query_end = cache_len + q_seq_len

            # Populate all image spans before applying per-image text deltas,
            # matching the two-loop ordering of the original Triton kernel.
            for image in images:
                image_start = int(image["start_idx"])
                image_len = int(image["token_num"])
                grid_thwd = image["grid_thwd"]
                image_width = int(grid_thwd[2])
                if image_width <= 0:
                    raise ValueError(f"invalid Neo image grid width: {image_width}")

                visible_start = max(image_start, cache_len)
                visible_end = min(image_start + image_len, query_end)
                if visible_start >= visible_end:
                    continue

                packed_image_start = packed_start + visible_start - cache_len
                packed_image_end = packed_start + visible_end - cache_len
                image_offsets = torch.arange(
                    visible_start - image_start,
                    visible_end - image_start,
                    dtype=position_ids.dtype,
                    device=position_ids.device,
                )
                image_slice = slice(packed_image_start, packed_image_end)
                position_ids[0, image_slice] = image_start
                position_ids[1, image_slice] = torch.div(image_offsets, image_width, rounding_mode="floor")
                position_ids[2, image_slice] = torch.remainder(image_offsets, image_width)
                self.b_image_token_end[image_slice] = image_start + image_len

            for image in images:
                image_end = int(image["start_idx"]) + int(image["token_num"])
                delta = int(image["grid_thwd"][3])
                visible_text_start = max(image_end, cache_len)
                if visible_text_start >= query_end or delta == 0:
                    continue
                packed_text_start = packed_start + visible_text_start - cache_len
                packed_text_end = packed_start + q_seq_len
                position_ids[0, packed_text_start:packed_text_end].add_(delta)

        return position_ids.contiguous()
