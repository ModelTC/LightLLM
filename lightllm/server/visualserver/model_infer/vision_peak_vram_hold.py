"""Hold vision-model peak VRAM when co-located with the LLM.

On startup, each supported visual model runs a worst-case dummy forward, then keeps
the CUDA allocator high-water mark by not calling empty_cache(). The co-located LLM
profiles KV capacity via mem_get_info and therefore leaves headroom for vision peak
VRAM usage. If the dummy forward OOMs, startup fails immediately.
"""

import math
from typing import List, Tuple

import torch

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

_PEAK_VRAM_HOLD_OOM_HINT = (
    "Vision peak VRAM hold probe hit OOM. Lower --visual_infer_batch_size, "
    "--max_image_pixels, or --max_image_token_count, or place the vision model on a "
    "separate GPU with --visual_gpu_ids."
)


def _is_qwen_vl_model(model) -> bool:
    return all(hasattr(model, attr) for attr in ("patch_size", "spatial_merge_size", "in_channels", "temporal_patch_size"))


def _is_internvl_model(model) -> bool:
    return all(hasattr(model, attr) for attr in ("MAX_PATH_NUM", "IMAGE_H", "IMAGE_W"))


class VisionPeakVramHolder:
    """Hold peak vision VRAM at startup via a worst-case dummy forward (co-located LLM)."""

    def __init__(self, model):
        self.model = model

    @classmethod
    def supports(cls, model) -> bool:
        return _is_qwen_vl_model(model) or _is_internvl_model(model)

    @torch.no_grad()
    def hold(
        self, device_id: int, batch_size: int, max_image_pixels: int, max_image_token_count: int
    ) -> int:
        torch.cuda.set_device(device_id)
        baseline_reserved = torch.cuda.memory_reserved(device_id)
        torch.cuda.reset_peak_memory_stats(device_id)
        try:
            dummy = self._build_worst_case_input(batch_size, max_image_pixels, max_image_token_count)
            out = self.model.forward(**dummy)
            del out, dummy
        except (RuntimeError, torch.OutOfMemoryError) as e:
            logger.exception(str(e))
            raise Exception(_PEAK_VRAM_HOLD_OOM_HINT)
        peak_reserved = torch.cuda.max_memory_reserved(device_id)
        return int(max(0, peak_reserved - baseline_reserved))

    def _build_worst_case_input(self, batch_size, max_image_pixels, max_image_token_count) -> dict:
        if _is_qwen_vl_model(self.model):
            return self._build_qwen_vl_worst_case_input(batch_size, max_image_pixels, max_image_token_count)
        if _is_internvl_model(self.model):
            return self._build_internvl_worst_case_input(batch_size)
        raise NotImplementedError(f"Unsupported vision model for peak VRAM hold: {type(self.model)}")

    def _build_internvl_worst_case_input(self, batch_size) -> dict:
        num_tiles = int(self.model.MAX_PATH_NUM) * int(batch_size)
        dummy_images = torch.randn(
            (num_tiles, 3, self.model.IMAGE_H, self.model.IMAGE_W),
            dtype=self.model.data_type,
            device="cuda",
        )
        return {"pixel_values": dummy_images}

    def _build_qwen_vl_worst_case_input(self, batch_size, max_image_pixels, max_image_token_count) -> dict:
        (total_patches, row_width), grid_thw = self._compute_qwen_worst_case_grid(
            batch_size=batch_size,
            max_image_pixels=max_image_pixels,
            max_image_token_count=max_image_token_count,
            patch_size=self.model.patch_size,
            temporal_patch_size=self.model.temporal_patch_size,
            in_channels=self.model.in_channels,
            spatial_merge_size=self.model.spatial_merge_size,
        )
        dtype = next(self.model.parameters()).dtype
        hidden_states = torch.randn((total_patches, row_width), dtype=dtype, device="cuda")
        grid_thw_t = torch.tensor(grid_thw, dtype=torch.long, device="cuda")
        return {"hidden_states": hidden_states, "grid_thw": grid_thw_t}

    @staticmethod
    def _compute_qwen_worst_case_grid(
        batch_size: int,
        max_image_pixels: int,
        max_image_token_count: int,
        patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        spatial_merge_size: int,
    ) -> Tuple[Tuple[int, int], List[List[int]]]:
        spatial_merge_unit = spatial_merge_size * spatial_merge_size
        patches_by_tokens = max_image_token_count * spatial_merge_unit
        patches_by_pixels = max_image_pixels // (patch_size * patch_size)
        max_patches = max(1, min(patches_by_tokens, patches_by_pixels))

        side = int(math.isqrt(max_patches))
        if side * side < max_patches:
            side += 1
        if side % spatial_merge_size:
            side += spatial_merge_size - (side % spatial_merge_size)
        side = max(side, spatial_merge_size)

        grid_h = grid_w = side
        row_width = in_channels * temporal_patch_size * patch_size * patch_size
        total_patches = grid_h * grid_w * batch_size
        grid_thw = [[1, grid_h, grid_w] for _ in range(batch_size)]
        return (total_patches, row_width), grid_thw
