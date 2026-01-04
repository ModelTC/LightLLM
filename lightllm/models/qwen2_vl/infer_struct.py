from typing import Optional, List
import torch
import numpy as np
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.qwen2_vl.triton_kernel.get_mrope_position_ids import get_mrope_position_triton
from lightllm.models.llama.flashattention_infer_struct import FlashAttentionStateInfo
from lightllm.utils.envs_utils import get_env_start_args


class Qwen2VLInferStateInfo(LlamaInferStateInfo):
    init_flash_attention_state_func = FlashAttentionStateInfo._init_flash_attention_state

    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None

    def init_some_extra_state(self, model):
        rope_scaling = model.config.get("rope_scaling", {})
        self.rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))
        InferStateInfo.init_some_extra_state(self, model)
        if self.is_prefill:
            self.position_ids = self.get_mrope_position(self.multimodal_params)
        else:
            b_position_delta = [0 for _ in range(self.b_seq_len.shape[0])]
            for batch_idx, p in enumerate(self.multimodal_params):
                position_delta = 0
                for image in p["images"]:
                    position_delta += image["grid_thwd"][3]
                b_position_delta[batch_idx] = position_delta
            position_ids = self.position_ids + torch.tensor(b_position_delta, device=self.position_ids.device)
            self.position_ids = position_ids.unsqueeze(0).expand(3, -1)

        self.position_ids = self.position_ids.contiguous()
        self.position_cos = model._cos_cached[self.position_ids]
        self.position_sin = model._sin_cached[self.position_ids]
        if get_env_start_args().enable_fa3:
            self.max_seq_len = self.max_kv_seq_len
            self.q_max_seq_len = self.max_q_seq_len
            self.init_flash_attention_state_func(model)
        return

    def get_mrope_position(self, multimodal_params: List[dict]) -> torch.Tensor:
        if len(multimodal_params) == 0:
            return self.position_ids.unsqueeze(0).expand(3, -1)
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
                b_image_len.append(img["token_num"])
                b_image_thwd.append(img["grid_thwd"])
            b_image_nums.append(len(images))
            b_image_start_num.append(image_start_num)
            image_start_num += len(images)

        # 没有任何图片
        if image_start_num == 0:
            return self.position_ids.unsqueeze(0).expand(3, -1).contiguous()
        b_image_start_idx = torch.tensor(b_image_start_idx, device="cpu").cuda(non_blocking=True)
        b_image_thwd = torch.tensor(b_image_thwd, device="cpu").cuda(non_blocking=True)  # image_num x 4
        b_image_nums = torch.tensor(b_image_nums, device="cpu").cuda(non_blocking=True)
        b_image_start_num = torch.tensor(b_image_start_num, device="cpu").cuda(non_blocking=True)
        b_image_len = torch.tensor(b_image_len, device="cpu").cuda(non_blocking=True)
        position_ids = self.position_ids.unsqueeze(0).expand(3, -1).contiguous()
        get_mrope_position_triton(
            b_image_start_idx=b_image_start_idx,
            b_image_thwd=b_image_thwd,
            b_image_nums=b_image_nums,
            b_image_start_num=b_image_start_num,
            b_image_len=b_image_len,
            position_ids=position_ids,
            b_ready_cache_len=self.b_ready_cache_len,
            b_q_seq_len=self.b_q_seq_len,
            b_start_loc=self.b_start_loc,
        )
        return position_ids
