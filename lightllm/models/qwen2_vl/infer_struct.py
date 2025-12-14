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

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        rope_scaling = model.config.get("rope_scaling", {})
        self.rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))
        self.vision_config = model.config.get("vision_config", {})
        InferStateInfo.init_some_extra_state(self, model, input_ids)
        if self.is_prefill:
            position_ids = self.get_mrope_position(self.multimodal_params)
        else:
            b_position_delta = [0 for _ in range(self.b_seq_len.shape[0])]
            for batch_idx, p in enumerate(self.multimodal_params):
                position_delta = 0
                for image in p["images"]:
                    position_delta += image["grid_thwd"][3]
                b_position_delta[batch_idx] = position_delta
            position_ids = self.position_ids + torch.tensor(b_position_delta, device=self.position_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(3, -1)
        torch.set_printoptions(profile="full")
        print(f"position_ids is {position_ids}")
        self.position_cos = model._cos_cached[position_ids.unsqueeze(1)]  # (3, 1, L, D)
        self.position_sin = model._sin_cached[position_ids.unsqueeze(1)]  # (3, 1, L, D)
        if get_env_start_args().enable_fa3:
            self.max_seq_len = self.max_kv_seq_len
            self.q_max_seq_len = self.max_q_seq_len
            self.init_flash_attention_state_func(model, input_ids)
        return

    def get_image_position_ids(self, image_grid_thw: List[int]) -> torch.Tensor:
        # TODO: 跟 img embed 一样，缓存在shm里。

        spatial_merge_size = self.vision_config.get("spatial_merge_size", 2)
        tokens_per_second = self.vision_config.get("tokens_per_second", 1.0)
        t, h, w = image_grid_thw
        video_second_per_grid_t = 0.0

        llm_grid_t, llm_grid_h, llm_grid_w = (
            t,
            h // spatial_merge_size,
            w // spatial_merge_size,
        )
        t_index = (
            (
                torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w)
                * video_second_per_grid_t
                * tokens_per_second
            )
            .long()
            .flatten()
        )

        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
        return torch.stack([t_index, h_index, w_index]).reshape(3, -1)

    def get_mrope_position(self, multimodal_params: List[dict]) -> torch.Tensor:
        if len(multimodal_params) == 0:
            return self.position_ids.unsqueeze(0).expand(3, -1)
        b_image_start_idx = []
        b_image_pos_delta = []
        b_image_nums = []
        b_image_position_id = []
        b_image_start_num = []
        b_image_cu_len = []
        b_image_len = []
        image_start_num = 0
        image_cu_len = 0
        for _, p in enumerate(multimodal_params):
            for img in p["images"]:
                b_image_start_idx.append(img["start_idx"])
                b_image_len.append(img["token_num"])
                b_image_pos_delta.append(img["grid_thwd"][3])
                b_image_position_id.append(self.get_image_position_ids(img["grid_thwd"][:3]))
                b_image_cu_len.append(image_cu_len)
                image_cu_len += img["token_num"]
            b_image_nums.append(len(p["images"]))
            b_image_start_num.append(image_start_num)
            image_start_num += len(p["images"])
        # 没有任何图片
        if image_start_num == 0:
            return self.position_ids.unsqueeze(0).expand(3, -1).contiguous()

        b_image_start_idx = torch.tensor(b_image_start_idx, device=self.position_ids.device)
        b_image_pos_delta = torch.tensor(b_image_pos_delta, device=self.position_ids.device)
        b_image_nums = torch.tensor(b_image_nums, device=self.position_ids.device)
        b_image_start_num = torch.tensor(b_image_start_num, device=self.position_ids.device)
        b_image_cu_len = torch.tensor(b_image_cu_len, device=self.position_ids.device)
        b_image_len = torch.tensor(b_image_len, device=self.position_ids.device)
        b_image_position_id = torch.cat(b_image_position_id, dim=1).cuda(non_blocking=True)
        position_ids = self.position_ids.unsqueeze(0).expand(3, -1).contiguous()
        print(f"BFR get_mrope_position_triton {position_ids}")
        get_mrope_position_triton(
            b_image_start_idx=b_image_start_idx,
            b_image_pos_delta=b_image_pos_delta,
            b_image_nums=b_image_nums,
            b_image_start_num=b_image_start_num,
            b_image_len=b_image_len,
            b_image_cu_len=b_image_cu_len,
            b_image_position_id=b_image_position_id,
            position_ids=position_ids,
            b_ready_cache_len=self.b_ready_cache_len,
            b_seq_len=self.b_q_seq_len,
            b_start_loc=self.b_start_loc,
        )
        return position_ids
