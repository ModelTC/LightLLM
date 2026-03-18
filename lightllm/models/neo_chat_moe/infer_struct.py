from typing import Optional, List
import torch
import numpy as np
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.req_manager import ReqManager
from lightllm.models.neo_chat_moe.triton_kernel.get_neo_position import get_neo_position_triton
from lightllm.models.llama.model import LlamaTpPartModel


class NeoChatInferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None
        self.position_cos_h = None
        self.position_sin_h = None
        self.position_cos_w = None
        self.position_sin_w = None

    def init_some_extra_state(self, model: LlamaTpPartModel):
        LlamaInferStateInfo.init_some_extra_state(self, model)
        if self.is_prefill:
            self.b_image_token_tag = torch.zeros([self.position_ids.size(0)], dtype=torch.bool, device="cpu").cuda(
                non_blocking=True
            )
            self.position_ids = self.get_neo_position(self.multimodal_params)
        else:
            b_position_delta = [0 for _ in range(self.b_seq_len.shape[0])]
            for batch_idx, p in enumerate(self.multimodal_params):
                position_delta = 0
                for image in p["images"]:
                    position_delta += image["grid_thwd"][3]
                b_position_delta[batch_idx] = position_delta
            position_ids = self.position_ids + torch.tensor(b_position_delta, device=self.position_ids.device)
            self.position_ids = position_ids.unsqueeze(0).expand(3, -1).clone()
            self.position_ids[1:].zero_()

        self.position_ids = self.position_ids.contiguous()
        self.position_cos = model._cos_cached[self.position_ids[0]]
        self.position_sin = model._sin_cached[self.position_ids[0]]
        self.position_cos_h = model._hw_cos_cached[self.position_ids[1]]
        self.position_sin_h = model._hw_sin_cached[self.position_ids[1]]
        self.position_cos_w = model._hw_cos_cached[self.position_ids[2]]
        self.position_sin_w = model._hw_sin_cached[self.position_ids[2]]
        return

    def get_neo_position(self, multimodal_params: List[dict]) -> torch.Tensor:
        if len(multimodal_params) == 0:
            position_ids = self.position_ids.new_zeros((3, self.position_ids.size(0)))
            position_ids[0].copy_(self.position_ids)
            return position_ids
        b_image_start_idx = []
        b_image_nums = []
        b_image_start_num = []
        b_image_len = []
        image_start_num = 0
        b_image_thwd = []

        # pad multimodal_params to batch size.
        batch_size = self.b_q_seq_len.shape[0]
        multimodal_params = multimodal_params + [
            {"images": [], "audios": []} for _ in range(batch_size - len(multimodal_params))
        ]

        for _, p in enumerate(multimodal_params):
            images = p.get("images", [])
            for img in images:
                b_image_start_idx.append(img["start_idx"])
                a = img["start_idx"]
                print(f"img start_idx: {a}")
                b_image_len.append(img["token_num"])
                b_image_thwd.append(img["grid_thwd"])
            b_image_nums.append(len(images))
            b_image_start_num.append(image_start_num)
            image_start_num += len(images)

        # 没有任何图片
        if image_start_num == 0:
            position_ids = self.position_ids.new_zeros((3, self.position_ids.size(0)))
            position_ids[0].copy_(self.position_ids)
            return position_ids.contiguous()
        b_image_start_idx = torch.tensor(b_image_start_idx, device="cpu").cuda(non_blocking=True)
        b_image_thwd = torch.tensor(b_image_thwd, device="cpu").cuda(non_blocking=True)  # image_num x 4
        b_image_nums = torch.tensor(b_image_nums, device="cpu").cuda(non_blocking=True)
        b_image_start_num = torch.tensor(b_image_start_num, device="cpu").cuda(non_blocking=True)
        b_image_len = torch.tensor(b_image_len, device="cpu").cuda(non_blocking=True)

        position_ids = self.position_ids.new_zeros((3, self.position_ids.size(0)))
        position_ids[0].copy_(self.position_ids)

        get_neo_position_triton(
            b_image_start_idx=b_image_start_idx,
            b_image_thwd=b_image_thwd,
            b_image_nums=b_image_nums,
            b_image_start_num=b_image_start_num,
            b_image_len=b_image_len,
            position_ids=position_ids,
            b_ready_cache_len=self.b_ready_cache_len,
            b_q_seq_len=self.b_q_seq_len,
            b_start_loc=self.b_q_start_loc,
            b_image_token_tag=self.b_image_token_tag,
        )
        return position_ids
