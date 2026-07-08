import math
import torch
from typing import List, Tuple
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

_RESERVE_OOM_HINT = (
    "ViT worst-case activation reservation hit OOM. Lower --visual_infer_batch_size, "
    "--max_image_pixels, or --max_image_token_count, or place the ViT on a separate GPU "
    "with --visual_gpu_ids."
)


class WorstCaseReserveMixin:
    """Adds a reserve-and-HOLD worst-case activation probe to a visual model.

    Subclasses MUST implement build_worst_case_input(...). The reservation is held by
    deliberately NOT calling torch.cuda.empty_cache() — the retained allocator high-water
    mark is what the LLM router observes via mem_get_info during KV-pool profiling.
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
        # Baseline = memory already reserved by the loaded ViT weights. We return the activation
        # growth ABOVE this baseline so the published/logged value is the tunable activation
        # headroom (what --visual_infer_batch_size / --max_image_* control), not weights+activation.
        # The physical hold is unaffected: we still never empty_cache, so the full peak stays
        # reserved and visible to the LLM's mem_get_info profiling.
        baseline_reserved = torch.cuda.memory_reserved(device_id)
        torch.cuda.reset_peak_memory_stats(device_id)
        try:
            dummy = self.build_worst_case_input(batch_size, max_image_pixels, max_image_token_count)
            out = self.run_worst_case_forward(dummy)
            del out, dummy
        except (RuntimeError, torch.OutOfMemoryError) as e:
            logger.exception(str(e))
            raise Exception(_RESERVE_OOM_HINT)
        # NB: intentionally NO torch.cuda.empty_cache() here — holding the high-water mark IS the mechanism.
        peak_reserved = torch.cuda.max_memory_reserved(device_id)
        return int(max(0, peak_reserved - baseline_reserved))


class QwenVLWorstCaseMixin(WorstCaseReserveMixin):
    """Worst-case builder for Qwen2/2.5/3-VL visual towers (shared forward(hidden_states, grid_thw))."""

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
        """Pure shape math for the Qwen-VL worst case.

        Returns ((total_patches, row_width), grid_thw) where pixel_values has shape
        (total_patches, row_width) and grid_thw is one [t, h, w] triple per dummy image.
        Bounds each image by BOTH the per-image token cap and pixel cap (whichever is tighter),
        using the smallest square grid (sides multiples of spatial_merge_size) whose patch count
        is >= that cap. The side is rounded UP so the probe is an upper bound on the largest valid
        request and never under-reserves (a square floor could undershoot, e.g. isqrt(32768)=181
        -> 180x180 = 32400 patches < the 32768-patch cap).

        Assumes valid inputs (max_image_token_count > 0 and max_image_pixels >= (patch_size *
        spatial_merge_size)**2); smaller budgets are clamped up to a single spatial_merge_size tile.
        """
        spatial_merge_unit = spatial_merge_size * spatial_merge_size
        patches_by_tokens = max_image_token_count * spatial_merge_unit
        patches_by_pixels = max_image_pixels // (patch_size * patch_size)
        max_patches = max(1, min(patches_by_tokens, patches_by_pixels))

        side = int(math.isqrt(max_patches))
        if side * side < max_patches:
            side += 1  # ceil(sqrt) so side*side >= max_patches (never undershoot)
        if side % spatial_merge_size:
            side += spatial_merge_size - (side % spatial_merge_size)  # round up to a merge-unit multiple
        side = max(side, spatial_merge_size)  # never smaller than one merge unit

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
        # Derive dtype from the loaded weights rather than self.data_type — the latter is not
        # guaranteed to be a torch.dtype on every Qwen visual class; parameters() always is.
        dtype = next(self.parameters()).dtype
        hidden_states = torch.randn((total_patches, row_width), dtype=dtype, device="cuda")
        grid_thw_t = torch.tensor(grid_thw, dtype=torch.long, device="cuda")
        return {"hidden_states": hidden_states, "grid_thw": grid_thw_t}
