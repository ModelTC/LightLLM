from typing import Optional, List
import torch
import numpy as np
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.qwen2_vl.triton_kernel.get_mrope_position_ids import get_mrope_position_triton
from lightllm.utils.envs_utils import get_env_start_args


class Qwen2VLInferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None

    def init_some_extra_state(self, model):
        rope_scaling = model.config.get("rope_scaling", {})
        self.rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))
        InferStateInfo.init_some_extra_state(self, model)

        # Prefill builds complete 3-axis MRoPE positions from the prompt's
        # image/video layout. Request-level position deltas are decode-only.
        if self.is_prefill:
            assert self.b_position_delta is None, "prefill must not provide b_position_delta"
            self.position_ids = self.get_mrope_position(self.multimodal_params)

        # Decode base position_ids contains one scalar position
        # per request; add the cached multimodal delta and broadcast the result
        # to MRoPE's temporal/height/width axes.
        else:
            assert self.b_position_delta is not None, "decode requires b_position_delta"
            self._apply_mrope_position_delta()

        self.position_ids = self.position_ids.contiguous()
        self.position_cos = model._cos_cached[self.position_ids]
        self.position_sin = model._sin_cached[self.position_ids]
        return

    def _apply_mrope_position_delta(self):
        b_position_delta = self.b_position_delta.to(dtype=self.position_ids.dtype)
        assert b_position_delta.shape == self.position_ids.shape, (
            "b_position_delta must align with position_ids, "
            f"got delta_shape={b_position_delta.shape}, position_shape={self.position_ids.shape}"
        )
        position_ids = self.position_ids + b_position_delta
        self.position_ids = position_ids.unsqueeze(0).expand(3, -1)

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
            b_start_loc=self.b_q_start_loc,
        )
        return position_ids
