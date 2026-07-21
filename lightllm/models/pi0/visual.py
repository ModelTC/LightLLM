from __future__ import annotations

from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SiglipVisionConfig, SiglipVisionModel

from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight
from lightllm.common.quantization import Quantcfg
from lightllm.models.pi0.config import Pi0VLAConfig
from lightllm.models.pi0.layer_weights.loader import Pi0SafeTensorLoader
from lightllm.server.embed_cache.utils import get_shm_name_data, read_shm
from lightllm.server.multimodal_params import ImageItem


def resize_with_pad_torch(images: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """OpenPI inference resize: preserve aspect ratio and pad normalized images with -1."""
    added_batch = images.ndim == 3
    if added_batch:
        images = images.unsqueeze(0)
    channels_last = images.shape[-1] <= 4
    if channels_last:
        images = images.permute(0, 3, 1, 2)
    _, _, current_height, current_width = images.shape
    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = F.interpolate(
        images.float(),
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    if images.dtype == torch.uint8:
        resized = resized.round().clamp(0, 255).to(torch.uint8)
        padding_value = 0
    elif images.dtype.is_floating_point:
        resized = resized.clamp(-1.0, 1.0).to(images.dtype)
        padding_value = -1.0
    else:
        raise ValueError(f"unsupported image dtype: {images.dtype}")
    pad_top, extra_height = divmod(height - resized_height, 2)
    pad_bottom = pad_top + extra_height
    pad_left, extra_width = divmod(width - resized_width, 2)
    pad_right = pad_left + extra_width
    output = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=padding_value)
    if channels_last:
        output = output.permute(0, 2, 3, 1)
    if added_batch:
        output = output.squeeze(0)
    return output


def preprocess_image(image: torch.Tensor, resolution: tuple[int, int]) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.shape[-1] == 3:
        image = image.permute(0, 3, 1, 2)
    if image.shape[1] != 3:
        raise ValueError(f"expected RGB image tensor, got shape {tuple(image.shape)}")
    if image.dtype == torch.uint8:
        image = image.float() / 127.5 - 1.0
    elif not image.dtype.is_floating_point:
        raise ValueError(f"unsupported image dtype: {image.dtype}")
    else:
        image = image.float()
    if image.shape[-2:] != resolution:
        image = resize_with_pad_torch(image, *resolution)
    return image


class Pi0VisionEncoder:
    """SigLIP So400m/14 visual process component for pi0 and pi0.5."""

    CHECKPOINT_PREFIX = "paligemma_with_expert.paligemma.model.vision_tower."

    def __init__(
        self,
        config: Pi0VLAConfig,
        *,
        device: torch.device | str = "cuda",
        output_dtype: torch.dtype | None = None,
        quant_type: str = "none",
        quant_cfg_path: str | None = None,
    ):
        self.config = config
        self.device = torch.device(device)
        self.output_dtype = output_dtype or config.torch_dtype
        projector_name = "paligemma_with_expert.paligemma.model." "multi_modal_projector.linear.weight"
        projector_bias_name = projector_name.replace("weight", "bias")
        projector_quant = Quantcfg(
            {"n_layer": 1},
            quant_type=quant_type or "none",
            custom_cfg_path=quant_cfg_path,
        )
        self.projector = ROWMMWeight(
            in_dim=1152,
            out_dims=[config.vlm_hidden_size],
            weight_names=projector_name,
            data_type=torch.float32,
            bias_names=projector_bias_name,
            quant_method=projector_quant.get_quant_method(-1, "multi_modal_projector"),
            tp_rank=0,
            tp_world_size=1,
        )
        vision_config = SiglipVisionConfig(
            hidden_size=1152,
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
            num_channels=3,
            patch_size=14,
            image_size=config.image_resolution[0],
            attention_dropout=0.0,
            layer_norm_eps=1e-6,
            hidden_act="gelu_pytorch_tanh",
            projection_dim=config.vlm_hidden_size,
            vision_use_head=False,
        )
        vision_config._attn_implementation = "eager"
        with torch.device("meta"):
            self.vision_model = SiglipVisionModel(vision_config)

        with Pi0SafeTensorLoader(config.checkpoint_path) as loader:
            state = {}
            for key in loader.keys():
                if key.startswith(self.CHECKPOINT_PREFIX):
                    local_name = key[len(self.CHECKPOINT_PREFIX) :]
                    state[local_name] = loader.tensor(key, device=self.device, dtype=torch.float32)
            missing, unexpected = self.vision_model.load_state_dict(state, strict=False, assign=True)
            if missing or unexpected:
                raise RuntimeError(f"invalid pi0 vision checkpoint; missing={missing}, unexpected={unexpected}")
            # ``position_ids`` is a non-persistent buffer, so assign-based
            # loading does not materialize the value created under the meta
            # device context.  Leaving meta indices here makes embedding
            # lookup read undefined positions on CUDA.
            embeddings = self.vision_model.vision_model.embeddings
            embeddings.position_ids = torch.arange(
                embeddings.num_patches,
                dtype=torch.long,
                device=self.device,
            ).expand(1, -1)
            projector_weight = loader.tensor(projector_name, device="cpu")
            self.projector.load_hf_weights({projector_name: projector_weight})
            del projector_weight
            projector_bias = loader.tensor(projector_bias_name, device="cpu")
            self.projector.load_hf_weights({projector_bias_name: projector_bias})
            del projector_bias
            if not self.projector.verify_load():
                raise RuntimeError("pi0 multimodal projector failed to load")
        self.vision_model.eval()

    @torch.no_grad()
    def encode(self, images: torch.Tensor, *, preprocessed: bool = False) -> torch.Tensor:
        if not preprocessed:
            images = preprocess_image(images, self.config.image_resolution)
        images = images.to(device=self.device, dtype=torch.float32)
        hidden_states = self.vision_model(pixel_values=images).last_hidden_state
        # OpenPI's SigLIP head is a raw 1152->2048 projection. There is no
        # PaliGemma input rescaling on this split inference path.
        features = self.projector.mm(
            hidden_states.reshape(-1, hidden_states.shape[-1]),
            use_custom_tensor_mananger=False,
        ).reshape(*hidden_states.shape[:-1], self.config.vlm_hidden_size)
        return features.to(self.output_dtype)


class Pi0VisionModel:
    """Adapter that lets the regular visualserver own π0 image inference."""

    def __init__(self, model_kvargs: dict):
        self.model_kvargs = model_kvargs
        self.encoder: Pi0VisionEncoder | None = None

    def load_model(self, weight_dir: str):
        config = Pi0VLAConfig.from_model_dir(weight_dir, dtype=self.model_kvargs.get("data_type"))
        self.encoder = Pi0VisionEncoder(
            config,
            device=f"cuda:{torch.cuda.current_device()}",
            output_dtype=config.torch_dtype,
            quant_type=self.model_kvargs.get("quant_type") or "none",
            quant_cfg_path=self.model_kvargs.get("quant_cfg"),
        )
        return self

    def cuda(self):
        # Pi0VisionEncoder materializes directly on the visualserver device.
        return self

    def eval(self):
        return self

    @torch.no_grad()
    def encode(self, images: list[ImageItem]):
        if self.encoder is None:
            raise RuntimeError("pi0 vision model is not loaded")

        image_tensors = []
        uuids = []
        for image in images:
            if not isinstance(image, ImageItem):
                raise TypeError(f"unsupported pi0 visual input: {type(image)!r}")
            raw = read_shm(get_shm_name_data(image.uuid))
            array = torch.from_numpy(np.array(Image.open(BytesIO(raw)).convert("RGB"), copy=True))
            image_tensors.append(preprocess_image(array, self.encoder.config.image_resolution))
            uuids.append(image.uuid)

        if not image_tensors:
            return None
        pixels = torch.cat(image_tensors, dim=0)
        embeddings = self.encoder.encode(pixels, preprocessed=True)
        tokens_per_image = embeddings.shape[1]
        flattened = embeddings.reshape(-1, embeddings.shape[-1])
        valid_ids = [[index * tokens_per_image, (index + 1) * tokens_per_image] for index in range(len(images))]
        return flattened, uuids, valid_ids
