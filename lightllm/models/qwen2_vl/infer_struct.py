from typing import Optional
import torch
import numpy as np
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.qwen2_vl.triton_kernel.get_mrope_position_ids import gen_mrope_pos_triton


class Qwen2VLInferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None

    def build_mrope_position_ids(
        self,
        input_ids: torch.Tensor,
        position_ids_1d: torch.Tensor,
        spatial_merge_size: int = 2,
    ) -> torch.Tensor:
        device = input_ids.device
        total_L = input_ids.shape[0]

        assert position_ids_1d.shape[0] == total_L

        all_pos = torch.empty(3, total_L, device=device, dtype=torch.long)

        for b in range(self.batch_size):
            seq_start = int(self.b_start_loc[b].item())
            seq_len = int(self.b_q_seq_len[b].item())
            seq_end = seq_start + seq_len

            seq_pos1d = position_ids_1d[seq_start:seq_end]

            if not self.multimodal_params:
                return seq_pos1d.view(1, -1).expand(3, -1)

            seq_input_ids = input_ids[seq_start:seq_end]

            images = self.multimodal_params[b]["images"]

            pos3d = gen_mrope_pos_triton(
                seq_input_ids,
                seq_pos1d,
                images,
                spatial_merge_size,
            )  # (3, seq_len)

            all_pos[:, seq_start:seq_end] = pos3d

        return all_pos

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        rope_scaling = model.config.get("rope_scaling", {})
        self.rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))
        vision_config = model.config.get("vision_config", {})
        spatial_merge_size = vision_config.get("spatial_merge_size", 2)  # image_processor可能会传另一个值？
        if self.rope_type != "mrope":
            super().init_some_extra_state(model, input_ids)
            return
        InferStateInfo.init_some_extra_state(self, model, input_ids)
        if self.is_prefill:
            position_ids = self.build_mrope_position_ids(
                input_ids,
                self.position_ids,
                spatial_merge_size,
            )
            position_ids = position_ids.unsqueeze(1)
            model.last_pos = position_ids[..., -1:]
        else:
            model.last_pos += 1
            position_ids = model.last_pos
        torch.set_printoptions(profile="full")
        print(f"position_ids is {position_ids}")
        pos = position_ids
        self.position_cos = model._cos_cached[pos]  # (3, 1, L, D)
        self.position_sin = model._sin_cached[pos]  # (3, 1, L, D)
        return position_ids[..., -1:]
