import torch
import torch.nn.functional as F

from lightllm.models.qwen3_vl.qwen3_visual import Qwen3VisionTransformerPretrainedModel
from lightllm.models.qwen2_vl.qwen2_visual import VisionRotaryEmbedding


class Qwen4ExpVisionTransformer(Qwen3VisionTransformerPretrainedModel):
    def __init__(self, kvargs, **config):
        config = dict(config)
        if config.get("deepstack_visual_indexes"):
            raise ValueError("Qwen4Exp vision does not define deepstack outputs")
        config["deepstack_visual_indexes"] = []
        rope = config.get("rope_parameters") or {"rope_type": "axial", "rope_theta": 10000.0}
        if rope.get("rope_type", "axial") != "axial":
            raise ValueError("Qwen4Exp vision requires axial rotary embeddings")
        super().__init__(kvargs, **config)
        self.rotary_pos_emb = VisionRotaryEmbedding(
            self.hidden_size // self.num_heads // 2, theta=rope.get("rope_theta", 10000.0)
        ).cuda()

    def fast_pos_embed_interpolate(self, grid_thw):
        table = self.pos_embed.weight.float().T.reshape(
            1, self.hidden_size, self.num_grid_per_side, self.num_grid_per_side
        )
        merge = self.spatial_merge_size
        patches = []
        for frames, height, width in grid_thw.detach().cpu().tolist():
            if min(frames, height, width) < 1 or height % merge or width % merge:
                raise ValueError("Vision grids must be positive and divisible by the spatial merge size")
            positions = F.interpolate(table, size=(height, width), mode="bilinear", align_corners=True)
            positions = positions[0].permute(1, 2, 0)
            positions = positions.reshape(height // merge, merge, width // merge, merge, self.hidden_size)
            positions = positions.permute(0, 2, 1, 3, 4).reshape(-1, self.hidden_size).repeat(frames, 1)
            patches.append(positions.to(self.pos_embed.weight.dtype))
        return torch.cat(patches)
