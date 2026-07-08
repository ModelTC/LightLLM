"""Hold vision-model peak activation memory when co-located with the LLM.

On startup, each supported visual model runs a worst-case dummy forward, then keeps
the CUDA allocator high-water mark by not calling empty_cache(). The co-located LLM
profiles KV capacity via mem_get_info and therefore leaves headroom for vision peak
activations. If the dummy forward OOMs, startup fails immediately.
"""

import math
from typing import List, Tuple

import torch

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

_PEAK_ACTIVATION_HOLD_OOM_HINT = (
    "Vision peak activation hold probe hit OOM. Lower --visual_infer_batch_size, "
    "--max_image_pixels, or --max_image_token_count, or place the vision model on a "
    "separate GPU with --visual_gpu_ids."
)


class WorstCaseReserveMixin:
    """Mixin that probes and holds peak vision activation memory at startup.

    Subclasses implement build_worst_case_input(...) to construct the largest valid
    dummy batch. Peak memory is held by not calling torch.cuda.empty_cache(); the
    co-located LLM observes the reduced free memory through mem_get_info.
    """

    def build_worst_case_input(self, batch_size, max_image_pixels, max_image_token_count) -> dict:
        raise NotImplementedError

    def run_worst_case_forward(self, dummy: dict):
        return self.forward(**dummy)

    @torch.no_grad()
    def reserve_worst_case_activation(
        self, device_id: int, batch_size: int, max_image_pixels: int, max_image_token_count: int
    ) -> int:
        torch.cuda.set_device(device_id)
        # Weights are already on device; measure only activation growth above that baseline.
        baseline_reserved = torch.cuda.memory_reserved(device_id)
        torch.cuda.reset_peak_memory_stats(device_id)
        try:
            dummy = self.build_worst_case_input(batch_size, max_image_pixels, max_image_token_count)
            out = self.run_worst_case_forward(dummy)
            del out, dummy
        except (RuntimeError, torch.OutOfMemoryError) as e:
            logger.exception(str(e))
            raise Exception(_PEAK_ACTIVATION_HOLD_OOM_HINT)
        # Keep the allocator peak — this is what holds activation memory for co-location.
        peak_reserved = torch.cuda.max_memory_reserved(device_id)
        return int(max(0, peak_reserved - baseline_reserved))


class QwenVLWorstCaseMixin(WorstCaseReserveMixin):
    """Peak-activation hold for Qwen2/2.5/3-VL towers (forward(hidden_states, grid_thw))."""

    def compute_qwen_worst_case_grid(
        self,
        batch_size: int,
        max_image_pixels: int,
        max_image_token_count: int,
        patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        spatial_merge_size: int,
    ) -> Tuple[Tuple[int, int], List[List[int]]]:
        """Compute dummy input shapes for the Qwen-VL peak activation hold probe.

        Returns ((total_patches, row_width), grid_thw) for hidden_states and grid_thw.
        Each image is bounded by both --max_image_token_count and --max_image_pixels;
        grid side length is rounded up to a spatial_merge_size multiple so the probe
        never under-estimates the largest valid request.
        """
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

    def build_worst_case_input(self, batch_size, max_image_pixels, max_image_token_count) -> dict:
        (total_patches, row_width), grid_thw = self.compute_qwen_worst_case_grid(
            batch_size=batch_size,
            max_image_pixels=max_image_pixels,
            max_image_token_count=max_image_token_count,
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=self.in_channels,
            spatial_merge_size=self.spatial_merge_size,
        )
        dtype = next(self.parameters()).dtype
        hidden_states = torch.randn((total_patches, row_width), dtype=dtype, device="cuda")
        grid_thw_t = torch.tensor(grid_thw, dtype=torch.long, device="cuda")
        return {"hidden_states": hidden_states, "grid_thw": grid_thw_t}
