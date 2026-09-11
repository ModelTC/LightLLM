from __future__ import annotations

from typing import List

import torch

from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.neo_chat.triton_kernel.get_neo_position import (
    get_neo_position_triton,
)


class NeoChatInferStateInfo(LlamaInferStateInfo):
    """Neo's three-axis position state for text/image understanding."""

    def __init__(self):
        super().__init__()
        self.position_cos_h = None
        self.position_sin_h = None
        self.position_cos_w = None
        self.position_sin_w = None
        self.b_image_token_end = None

    def init_some_extra_state(self, model):
        # Do not call LlamaInferStateInfo here: its one-axis RoPE table has a
        # different width.  First build only the common packed-sequence state.
        InferStateInfo.init_some_extra_state(self, model)
        if self.is_prefill:
            self.max_seq_len = self.max_kv_seq_len
            self.b_image_token_end = torch.zeros_like(
                self.position_ids,
                dtype=torch.int32,
                device=self.position_ids.device,
            )
            self.position_ids = self.get_neo_position(self.multimodal_params)
        else:
            delta = self.b_position_delta
            if delta is None:
                delta = torch.zeros_like(self.position_ids)
            temporal = self.position_ids + delta.to(self.position_ids.dtype)
            self.position_ids = temporal.unsqueeze(0).expand(3, -1).clone()
            self.position_ids[1:].zero_()

        self.position_ids = self.position_ids.contiguous()
        self.position_cos = model._cos_cached[self.position_ids[0]]
        self.position_sin = model._sin_cached[self.position_ids[0]]
        self.position_cos_h = model._hw_cos_cached[self.position_ids[1]]
        self.position_sin_h = model._hw_sin_cached[self.position_ids[1]]
        self.position_cos_w = model._hw_cos_cached[self.position_ids[2]]
        self.position_sin_w = model._hw_sin_cached[self.position_ids[2]]

    def get_neo_position(self, multimodal_params: List[dict]) -> torch.Tensor:
        multimodal_params = list(multimodal_params or ())
        batch_size = self.b_q_seq_len.shape[0]
        if len(multimodal_params) < batch_size:
            multimodal_params.extend(
                {"images": [], "audios": []}
                for _ in range(batch_size - len(multimodal_params))
            )

        image_start_indexes = []
        image_counts = []
        image_offsets = []
        image_lengths = []
        image_grids = []
        image_offset = 0
        for params in multimodal_params[:batch_size]:
            images = params.get("images", [])
            image_offsets.append(image_offset)
            image_counts.append(len(images))
            image_offset += len(images)
            for image in images:
                image_start_indexes.append(image["start_idx"])
                image_lengths.append(image["token_num"])
                image_grids.append(image["grid_thwd"])

        position_ids = self.position_ids.new_zeros((3, self.position_ids.numel()))
        position_ids[0].copy_(self.position_ids)
        if not image_grids:
            return position_ids.contiguous()

        def cuda_int(values):
            return torch.tensor(
                values,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            ).to(device=self.position_ids.device, non_blocking=True)

        get_neo_position_triton(
            b_image_start_idx=cuda_int(image_start_indexes),
            b_image_thwd=cuda_int(image_grids),
            b_image_nums=cuda_int(image_counts),
            b_image_start_num=cuda_int(image_offsets),
            b_image_len=cuda_int(image_lengths),
            position_ids=position_ids,
            b_ready_cache_len=self.b_ready_cache_len,
            b_q_seq_len=self.b_q_seq_len,
            b_start_loc=self.b_q_start_loc,
            b_image_token_end=self.b_image_token_end,
        )
        return position_ids
