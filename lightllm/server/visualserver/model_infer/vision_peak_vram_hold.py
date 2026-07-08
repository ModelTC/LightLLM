"""Hold vision-model peak VRAM when co-located with the LLM.

Build worst-case ImageItem inputs (max pixel budget, extreme aspect ratio, solid RGB),
run the normal encode() path, then keep the CUDA allocator high-water mark.
"""

import math
import uuid
from io import BytesIO
from typing import List, Tuple

import torch
from PIL import Image

from lightllm.server.embed_cache.utils import create_shm, free_shm, get_shm_name_data
from lightllm.server.multimodal_params import ImageItem
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

_PEAK_VRAM_HOLD_OOM_HINT = (
    "Vision peak VRAM hold probe hit OOM. Lower --visual_infer_batch_size, "
    "--max_image_pixels, or --max_image_token_count, or place the vision model on a "
    "separate GPU with --visual_gpu_ids."
)

# Match Qwen-VL smart_resize MAX_RATIO; extreme aspect ratio stresses vision sequence length.
_MAX_ASPECT_RATIO = 200


def _worst_case_image_size(max_image_pixels: int) -> Tuple[int, int]:
    """Largest valid landscape aspect ratio under the pixel budget (width >= height)."""
    height = max(1, int(math.sqrt(max_image_pixels / _MAX_ASPECT_RATIO)))
    width = min(max_image_pixels // height, height * _MAX_ASPECT_RATIO)
    height = int(max(1, height))
    width = int(max(1, width))
    return width, height


def _solid_rgb_jpeg_bytes(width: int, height: int, color: Tuple[int, int, int] = (0, 0, 0)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


class VisionPeakVramHolder:
    """Hold peak vision VRAM at startup via worst-case ImageItem encode()."""

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def hold(
        self, device_id: int, batch_size: int, max_image_pixels: int, max_image_token_count: int
    ) -> int:
        torch.cuda.set_device(device_id)
        baseline_reserved = torch.cuda.memory_reserved(device_id)
        torch.cuda.reset_peak_memory_stats(device_id)
        image_items = self._build_worst_case_image_items(
            batch_size, max_image_pixels, max_image_token_count
        )
        try:
            out = self.model.encode(image_items)
            del out
        except (RuntimeError, torch.OutOfMemoryError) as e:
            logger.exception(str(e))
            raise Exception(_PEAK_VRAM_HOLD_OOM_HINT)
        finally:
            self._free_image_items(image_items)
        peak_reserved = torch.cuda.max_memory_reserved(device_id)
        return int(max(0, peak_reserved - baseline_reserved))

    def _build_worst_case_image_items(
        self, batch_size: int, max_image_pixels: int, max_image_token_count: int
    ) -> List[ImageItem]:
        width, height = _worst_case_image_size(max_image_pixels)
        items = []
        for batch_id in range(batch_size):
            # Alternate solid black/white so each batch slot gets a distinct JPEG payload.
            color = (255, 255, 255) if batch_id % 2 else (0, 0, 0)
            image_bytes = _solid_rgb_jpeg_bytes(width, height, color=color)
            item = ImageItem(type="base64", data="")
            item.uuid = f"vision_peak_hold_{batch_id}_{uuid.uuid4().hex}"
            item.image_w = width
            item.image_h = height
            # InternVL encode() reads image_patch_max_num from extra_params (normally set by
            # InternvlTokenizer.init_imageitem_extral_params). Use MAX_PATH_NUM for worst-case tile count.
            if hasattr(self.model, "MAX_PATH_NUM"):
                item.extra_params["image_patch_max_num"] = int(self.model.MAX_PATH_NUM)

            create_shm(get_shm_name_data(item.uuid), image_bytes)
            items.append(item)
        return items

    @staticmethod
    def _free_image_items(items: List[ImageItem]) -> None:
        for item in items:
            if item.uuid is None:
                continue
            try:
                free_shm(get_shm_name_data(item.uuid))
            except FileNotFoundError:
                pass
            item.uuid = None
