from typing import List, Optional, Tuple, Union
import math
import torch.utils.checkpoint
from torch import nn
import transformers
import numpy as np
import base64
from PIL import Image
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.cache_utils import Cache, DynamicCache
from lightllm.server.core.objs.x2i_params import X2IParams
import torchvision.io as io

from .configuration_neo_chat import NEOChatConfig
from .modeling_neo_vit import NEOVisionModel
from .modeling_qwen3 import Qwen3ForCausalLM, create_block_causal_mask
from .modeling_fm_modules import PositionEmbedding, TimestepEmbedder, FlowMatchingHead, RMSNorm, NerfEmbedder, SimpleMLPAdaLN, ConvDecoder


logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

def prepare_flash_kv_cache(
    past_key_values,
    current_len: int,
    batch_size: int,
):
    """
    Convert prefix cache from [B, H, S, D] to flash-attn friendly [B, S, H, D],
    and preallocate full KV buffer for [prefix + current].

    This is done once before denoising loop.
    """
    if past_key_values is None:
        return

    for layer in past_key_values.layers:
        past_k = layer.keys
        past_v = layer.values

        if past_k is None or past_v is None:
            layer.flash_prefix_len = 0
            layer.flash_total_len = current_len
            layer.flash_k_cache = None
            layer.flash_v_cache = None
            continue

        # original cache layout assumed: [B, H, S, D]
        past_k_flash = past_k.transpose(1, 2).contiguous()  # [B, S, H, D]
        past_v_flash = past_v.transpose(1, 2).contiguous()  # [B, S, H, D]

        prefix_len = past_k_flash.shape[1]
        total_len = prefix_len + current_len

        k_cache = torch.empty(
            (batch_size, total_len, past_k_flash.shape[2], past_k_flash.shape[3]),
            device=past_k_flash.device,
            dtype=past_k_flash.dtype,
        )
        v_cache = torch.empty(
            (batch_size, total_len, past_v_flash.shape[2], past_v_flash.shape[3]),
            device=past_v_flash.device,
            dtype=past_v_flash.dtype,
        )

        k_cache[:, :prefix_len].copy_(past_k_flash)
        v_cache[:, :prefix_len].copy_(past_v_flash)

        layer.flash_prefix_len = prefix_len
        layer.flash_total_len = total_len
        layer.flash_k_cache = k_cache
        layer.flash_v_cache = v_cache

def clear_flash_kv_cache(past_key_values):
    if past_key_values is None:
        return
    for layer in past_key_values.layers:
        if hasattr(layer, "flash_prefix_len"):
            delattr(layer, "flash_prefix_len")
        if hasattr(layer, "flash_total_len"):
            delattr(layer, "flash_total_len")
        if hasattr(layer, "flash_k_cache"):
            delattr(layer, "flash_k_cache")
        if hasattr(layer, "flash_v_cache"):
            delattr(layer, "flash_v_cache")


@torch.cuda.amp.autocast(dtype=torch.float32)
def optimized_scale(positive_flat, negative_flat):

    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm

    return st_star


def build_abs_positions_from_grid_hw(grid_hw: torch.Tensor, device=None):
    """
    Compute patch coordinates (x, y)

    Args:
        grid_hw: (B, 2) tensor representing (H, W) per image
    """
    device = grid_hw.device
    B = grid_hw.shape[0]

    # Get the number of patches per image
    H = grid_hw[:, 0]
    W = grid_hw[:, 1]
    N = H * W
    N_total = N.sum()

    # Create the batch index for each patch (B x patch count)
    patch_to_sample = torch.repeat_interleave(torch.arange(B, device=device), N)  # (N_total,)

    # Generate intra-image patch index (row-major order)
    patch_id_within_image = torch.arange(N_total, device=device)
    patch_id_within_image = patch_id_within_image - torch.cumsum(
        torch.cat([torch.tensor([0], device=device), N[:-1]]), dim=0
    )[patch_to_sample]

    # Get H/W for each patch according to its image
    W_per_patch = W[patch_to_sample]
    abs_x = patch_id_within_image % W_per_patch
    abs_y = patch_id_within_image // W_per_patch

    return abs_x, abs_y


class NEOChatModel(PreTrainedModel):
    config_class = NEOChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "NEOVisionModel",
        "Qwen3DecoderLayer",
    ]

    # support transformers 4.51.+
    _tp_plan = ''

    def __init__(self, config: NEOChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.template = config.template
        self.downsample_ratio = config.downsample_ratio
        config.llm_config._attn_implementation = 'eager'

        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = NEOVisionModel(config.vision_config)
            vision_model_mot_gen = NEOVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            self.language_model = Qwen3ForCausalLM(config.llm_config)

        merge_size = int(1 / self.downsample_ratio)
        output_dim = 3*(patch_size*merge_size)**2
        llm_hidden_size = self.config.llm_config.hidden_size
        self.use_deep_fm_head = self.config.fm_head_layers > 2
        self.use_pixel_head = self.config.use_pixel_head

        if self.use_deep_fm_head:
                fm_head = FlowMatchingHead(llm_hidden_size, output_dim, dim=self.config.fm_head_dim, layers=self.config.fm_head_layers, mlp_ratio=self.config.fm_head_mlp_ratio)
        else:
            fm_head = nn.Sequential(
                    nn.Linear(llm_hidden_size, 4096, bias=True),
                    nn.GELU(),
                    nn.Linear(4096, output_dim, bias=True),
                )

        timestep_embedder = TimestepEmbedder(llm_hidden_size)
        self.fm_modules = nn.ModuleDict(
                    {
                        "vision_model_mot_gen": vision_model_mot_gen,
                        "timestep_embedder": timestep_embedder,
                        "fm_head": fm_head
                    }
                )
        if self.use_pixel_head:
            self.fm_modules["fm_head"] = ConvDecoder(llm_hidden_size)


        self.concat_time_token_num = config.concat_time_token_num
        self.time_token_id = 151682
        self.noise_scale = config.noise_scale
        self.noise_scale_mode = config.noise_scale_mode
        self.noise_scale_base_image_seq_len = config.noise_scale_base_image_seq_len

        self.add_noise_scale_embedding = config.add_noise_scale_embedding
        self.noise_scale_max_value = config.noise_scale_max_value
        self.time_schedule = config.time_schedule
        self.time_shift_type = config.time_shift_type
        self.base_shift = config.base_shift
        self.max_shift = config.max_shift
        self.base_image_seq_len = config.base_image_seq_len
        self.max_image_seq_len = config.max_image_seq_len

        if self.add_noise_scale_embedding:
            noise_scale_embedder = TimestepEmbedder(llm_hidden_size)
            self.fm_modules['noise_scale_embedder'] = noise_scale_embedder



        self.img_context_token_id = None
        self.img_start_token_id = 151670
        # self.conv_template = get_conv_template(self.template)
        # self.system_message = self.conv_template.system_message


    def extract_feature(self, pixel_values, gen_model=False, grid_hw=None):
        if gen_model:
            return self.fm_modules['vision_model_mot_gen'](pixel_values=pixel_values,
                                 output_hidden_states=False,
                                 return_dict=True,
                                 grid_hw=grid_hw).last_hidden_state
        else:
            return self.vision_model(pixel_values=pixel_values,
                                 output_hidden_states=False,
                                 return_dict=True,
                                 grid_hw=grid_hw).last_hidden_state

    def patchify(self, images, patch_size, channel_first=False):
        """
        images: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        h, w = images.shape[2] // patch_size, images.shape[3] // patch_size
        x = images.reshape(shape=(images.shape[0], 3, h, patch_size, w, patch_size))

        if channel_first:
            x = torch.einsum('nchpwq->nhwcpq', x)
        else:
            x = torch.einsum('nchpwq->nhwpqc', x)

        x = x.reshape(shape=(images.shape[0], h * w, patch_size**2 * 3))
        return x

    def unpatchify(sle, x, patch_size, h=None, w=None):
        """
        x: (N, L, patch_size**2 *3)
        images: (N, 3, H, W)
        """
        if h is None or w is None:
            h = w = int(x.shape[1]**.5)
        else:
            h = h // patch_size
            w = w // patch_size
        x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        images = x.reshape(shape=(x.shape[0], 3, h * patch_size, w * patch_size))
        return images

    def _euler_step(self, v_pred, z, t, t_next):
        z_next = z + (t_next - t) * v_pred
        return z_next

    def _calculate_dynamic_mu(self, image_seq_len: int) -> float:
        denom = self.max_image_seq_len - self.base_image_seq_len
        if denom == 0:
            return float(self.base_shift)
        m = (self.max_shift - self.base_shift) / denom
        b = self.base_shift - m * self.base_image_seq_len
        return float(image_seq_len) * m + b

    def _apply_time_schedule(self, t: torch.Tensor, image_seq_len: int, timestep_shift: float) -> torch.Tensor:
        self.time_schedule = "standard"
        sigma = 1 - t
        if timestep_shift != 1:
            self.time_schedule = "standard"
        if self.time_schedule == "standard":
            shift = timestep_shift
            sigma = shift * sigma / (1 + (shift - 1) * sigma)
        elif self.time_schedule == "dynamic":
            mu = self._calculate_dynamic_mu(image_seq_len)
            mu_t = t.new_tensor(mu)
            if self.time_shift_type == "exponential":
                shift = torch.exp(mu_t)
                sigma = shift * sigma / (1 + (shift - 1) * sigma)
            elif self.time_shift_type == "linear":
                sigma = mu_t / (mu_t + (1 / sigma - 1))
            else:
                raise ValueError(f"Unsupported time_shift_type: {self.time_shift_type}")
        else:
            raise ValueError(f"Unsupported time_schedule: {self.time_schedule}")
        return 1 - sigma

    def _build_t2i_image_indexes(self, token_h, token_w, text_len, device):
        t_image = torch.full((token_h * token_w,), text_len, dtype=torch.long, device=device)
        idx = torch.arange(token_h * token_w, device=device, dtype=torch.long)
        h_image = idx // token_w
        w_image = idx % token_w
        return torch.stack([t_image, h_image, w_image], dim=0)



    def _t2i_predict_v(self, input_embeds, indexes_image, attn_mask, past_key_values, t, z,
                       image_token_num, timestep_embeddings=None, image_size=None):
        B, L = z.shape[0], z.shape[1]

        outputs = self.language_model.model(
            inputs_embeds=input_embeds,
            image_gen_indicators=torch.ones((input_embeds.shape[0], input_embeds.shape[1]), dtype=torch.bool, device=input_embeds.device),
            indexes=indexes_image,
            attention_mask=attn_mask,
            past_key_values=past_key_values,
            update_cache=False,
            use_cache=True,
        )

        if self.use_pixel_head:
            merge_size = int(1 / self.downsample_ratio)
            token_h = image_size[1] // (self.patch_size * merge_size)
            token_w = image_size[0] // (self.patch_size * merge_size)

            img_reshaped = outputs.last_hidden_state[:, -image_token_num:].view(B, token_h, token_w, -1)
            img_2d = torch.einsum("b h w c -> b c h w", img_reshaped)
            img_2d = img_2d.contiguous().view(B, -1, token_h, token_w)

            smoothed_img_2d = self.fm_modules['fm_head'](img_2d)

            smoothed_reshaped = smoothed_img_2d.view(B, 3, token_h, self.patch_size * merge_size, token_w, self.patch_size * merge_size)
            smoothed_reshaped = torch.einsum("b c h p w q -> b h w p q c", smoothed_reshaped)
            out_1d = smoothed_reshaped.contiguous().view(B, L, self.patch_size * merge_size * self.patch_size * merge_size * 3)
            x_pred = out_1d
        else:
            if self.use_deep_fm_head:
                x_pred = self.fm_modules["fm_head"](
                outputs.last_hidden_state[:, -image_token_num:].view(B*L, -1), t.repeat(B*L)
                ).view(B, L, -1)
            else:
                x_pred = self.fm_modules["fm_head"](
                    outputs.last_hidden_state[:, -image_token_num:].view(B, L, -1)
                ).view(B, L, -1)


        v_pred = (x_pred - z) / (1 - t).clamp_min(self.config.t_eps)
        return v_pred


    @torch.no_grad()
    def it2i_generate(self,
                      past_key_values_condition,
                      past_key_values_text_uncondition,
                      past_key_values_img_uncondition,
                      text_lens,
                      cfg_scale=1,
                      img_cfg_scale=1,
                      cfg_norm='none',
                      enable_timestep_shift=True,
                      timestep_shift=1,
                      image_size=(256, 256),
                      num_steps=30,
                      cfg_interval=(0.1, 1.0),
                      batch_size=1,
                      t_eps=0.02,
                      ):

        self.config.t_eps = t_eps
        device, dtype = self.get_cache_device_dtype(past_key_values_condition)
        S1, S2, S3 = text_lens

        merge_size = int(1 / self.downsample_ratio)

        token_h = image_size[1] // (self.patch_size * merge_size)
        token_w = image_size[0] // (self.patch_size * merge_size)

        indexes_image_condition = self._build_t2i_image_indexes(token_h, token_w, S1, device=device)
        indexes_image_text_uncondition = self._build_t2i_image_indexes(token_h, token_w, S2, device=device)
        indexes_image_img_uncondition = self._build_t2i_image_indexes(token_h, token_w, S3, device=device)

        for layer_idx in range(len(past_key_values_condition.layers)):
            past_key_values_condition.layers[layer_idx].keys = past_key_values_condition.layers[layer_idx].keys.expand(batch_size, *past_key_values_condition.layers[layer_idx].keys.shape[1:])
            past_key_values_condition.layers[layer_idx].values = past_key_values_condition.layers[layer_idx].values.expand(batch_size, *past_key_values_condition.layers[layer_idx].values.shape[1:])
            past_key_values_text_uncondition.layers[layer_idx].keys = past_key_values_text_uncondition.layers[layer_idx].keys.expand(batch_size, *past_key_values_text_uncondition.layers[layer_idx].keys.shape[1:])
            past_key_values_text_uncondition.layers[layer_idx].values = past_key_values_text_uncondition.layers[layer_idx].values.expand(batch_size, *past_key_values_text_uncondition.layers[layer_idx].values.shape[1:])
            past_key_values_img_uncondition.layers[layer_idx].keys = past_key_values_img_uncondition.layers[layer_idx].keys.expand(batch_size, *past_key_values_img_uncondition.layers[layer_idx].keys.shape[1:])
            past_key_values_img_uncondition.layers[layer_idx].values = past_key_values_img_uncondition.layers[layer_idx].values.expand(batch_size, *past_key_values_img_uncondition.layers[layer_idx].values.shape[1:])

        prepare_flash_kv_cache(
            past_key_values_condition,
            current_len=token_h * token_w,
            batch_size=batch_size,
        )
        prepare_flash_kv_cache(
            past_key_values_text_uncondition,
            current_len=token_h * token_w,
            batch_size=batch_size,
        )
        prepare_flash_kv_cache(
            past_key_values_img_uncondition,
            current_len=token_h * token_w,
            batch_size=batch_size,
        )


        # init noise image tokens
        grid_h = image_size[1] // self.patch_size
        grid_w = image_size[0] // self.patch_size
        grid_hw = torch.tensor([[grid_h, grid_w]]*batch_size, device=device)

        noise_scale = self.noise_scale
        if self.noise_scale_mode in ("resolution", "dynamic", 'dynamic_sqrt'):
            noise_scale = math.sqrt((grid_h*grid_w)/(merge_size**2) / self.noise_scale_base_image_seq_len)
            base = float(self.noise_scale_base_image_seq_len)
            scale = math.sqrt((grid_h*grid_w)/(merge_size**2)/base)
            noise_scale = scale * float(self.noise_scale)
            if self.noise_scale_mode == 'dynamic_sqrt':
                noise_scale = math.sqrt(noise_scale)
        noise_scale = min(noise_scale, self.noise_scale_max_value)

        image_prediction = noise_scale * torch.randn((batch_size, 3, image_size[1], image_size[0]), device=device, dtype=dtype)

        attention_mask_condition = {"full_attention": None}
        attention_mask_text_uncondition = {"full_attention": None}
        attention_mask_img_uncondition = {"full_attention": None}

        timesteps = torch.linspace(0.0, 1.0, num_steps+1, device=device)
        if enable_timestep_shift:
            timesteps = self._apply_time_schedule(timesteps, token_h*token_w, timestep_shift)

        for step_i in range(num_steps):
            t = timesteps[step_i]
            t_next = timesteps[step_i + 1]

            z = self.patchify(image_prediction, self.patch_size * merge_size)
            image_input = self.patchify(image_prediction, self.patch_size, channel_first=True)
            image_embeds = self.extract_feature(image_input.view(batch_size * grid_h*grid_w, -1), gen_model=True, grid_hw=grid_hw).view(batch_size, token_h*token_w, -1)
            t_expanded = t.expand(batch_size*token_h*token_w)
            timestep_embeddings = self.fm_modules['timestep_embedder'](t_expanded).view(batch_size, token_h*token_w, -1)
            if self.add_noise_scale_embedding:
                noise_scale_tensor = torch.full_like(t_expanded, noise_scale/self.noise_scale_max_value)
                noise_embeddings = self.fm_modules['noise_scale_embedder'](noise_scale_tensor).view(batch_size, token_h*token_w, -1)
                timestep_embeddings += noise_embeddings
            image_embeds = image_embeds + timestep_embeddings

            v_pred_condition = self._t2i_predict_v(image_embeds, indexes_image_condition, attention_mask_condition, past_key_values_condition, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings,image_size=image_size)
            if t > cfg_interval[0] and t < cfg_interval[1]:
                if cfg_scale > 1:
                    v_pred_text_uncondition = self._t2i_predict_v(image_embeds, indexes_image_text_uncondition, attention_mask_text_uncondition, past_key_values_text_uncondition, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings,image_size=image_size)
                else:
                    v_pred_text_uncondition = 0
                if img_cfg_scale > 1:
                    v_pred_img_uncondition = self._t2i_predict_v(image_embeds, indexes_image_img_uncondition, attention_mask_img_uncondition, past_key_values_img_uncondition, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings,image_size=image_size)
                else:
                    v_pred_img_uncondition = 0

            if t > cfg_interval[0] and t < cfg_interval[1]:
                v_pred_text = v_pred_text_uncondition + cfg_scale * (v_pred_condition - v_pred_text_uncondition)
                if cfg_norm == 'text_channel':
                    norm_v_condition = torch.norm(v_pred_condition, dim=-1, keepdim=True)
                    norm_v_cfg = torch.norm(v_pred_text, dim=-1, keepdim=True)
                    scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                    v_pred_text = v_pred_text * scale
                v_pred = v_pred_img_uncondition + img_cfg_scale * (v_pred_text - v_pred_img_uncondition)
                if cfg_norm == 'global':
                    norm_v_condition = torch.norm(v_pred_condition, dim=(1,2), keepdim=True)
                    norm_v_cfg = torch.norm(v_pred, dim=(1,2), keepdim=True)
                    scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                    v_pred = v_pred * scale
                elif cfg_norm == 'channel':
                    norm_v_condition = torch.norm(v_pred_condition, dim=-1, keepdim=True)
                    norm_v_cfg = torch.norm(v_pred, dim=-1, keepdim=True)
                    scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                    v_pred = v_pred * scale

            else:
                v_pred = v_pred_condition

            z = z + (t_next - t) * v_pred

            image_prediction = self.unpatchify(z, self.patch_size * merge_size, image_size[1], image_size[0])

        clear_flash_kv_cache(past_key_values_condition)
        clear_flash_kv_cache(past_key_values_text_uncondition)
        clear_flash_kv_cache(past_key_values_img_uncondition)

        return image_prediction


    def get_cache_device_dtype(self, cache):
        """
        Returns (device, dtype) of a DynamicCache.
        Assumes all layers share same device/dtype.
        """
        for layer in cache.layers:
            return layer.device, layer.dtype
        raise ValueError("Cache is empty")

    @torch.no_grad()
    def t2i_generate(self,
                     past_key_values_condition,
                     past_key_values_uncondition,
                     text_lens,
                     cfg_scale=1,
                     timestep_shift=1,
                     enable_timestep_shift=True,
                     cfg_norm='none',
                     image_size=(256, 256),
                     num_steps=30,
                     cfg_interval=(0.1, 1.0),
                     batch_size=1,
                     t_eps=0.02):
        assert cfg_norm in ['cfg_zero_star', 'global', 'none'], f"cfg_norm={cfg_norm}"
        merge_size = int(1 / self.downsample_ratio)
        self.config.t_eps = t_eps

        token_h = image_size[1] // (self.patch_size * merge_size)
        token_w = image_size[0] // (self.patch_size * merge_size)

        device, dtype = self.get_cache_device_dtype(past_key_values_condition)
        S1, S2 = text_lens

        indexes_image_condition = self._build_t2i_image_indexes(token_h, token_w, S1, device=device)
        indexes_image_uncondition = self._build_t2i_image_indexes(token_h, token_w, S2, device=device)

        for layer_idx in range(len(past_key_values_condition.layers)):
            past_key_values_condition.layers[layer_idx].keys = past_key_values_condition.layers[layer_idx].keys.expand(batch_size, *past_key_values_condition.layers[layer_idx].keys.shape[1:])
            past_key_values_condition.layers[layer_idx].values = past_key_values_condition.layers[layer_idx].values.expand(batch_size, *past_key_values_condition.layers[layer_idx].values.shape[1:])
            past_key_values_uncondition.layers[layer_idx].keys = past_key_values_uncondition.layers[layer_idx].keys.expand(batch_size, *past_key_values_uncondition.layers[layer_idx].keys.shape[1:])
            past_key_values_uncondition.layers[layer_idx].values = past_key_values_uncondition.layers[layer_idx].values.expand(batch_size, *past_key_values_uncondition.layers[layer_idx].values.shape[1:])

        # prepare flash cache once
        prepare_flash_kv_cache(
            past_key_values_condition,
            current_len=token_h * token_w,
            batch_size=batch_size,
        )
        prepare_flash_kv_cache(
            past_key_values_uncondition,
            current_len=token_h * token_w,
            batch_size=batch_size,
        )

        # init noise image tokens
        grid_h = image_size[1] // self.patch_size
        grid_w = image_size[0] // self.patch_size
        grid_hw = torch.tensor([[grid_h, grid_w]]*batch_size, device=device)

        noise_scale = self.noise_scale
        if self.noise_scale_mode in ("resolution", "dynamic", 'dynamic_sqrt'):
            noise_scale = math.sqrt((grid_h*grid_w)/(merge_size**2) / self.noise_scale_base_image_seq_len)
            base = float(self.noise_scale_base_image_seq_len)
            scale = math.sqrt((grid_h*grid_w)/(merge_size**2)/base)
            noise_scale = scale * float(self.noise_scale)
            if self.noise_scale_mode == 'dynamic_sqrt':
                noise_scale = math.sqrt(noise_scale)
        noise_scale = min(noise_scale, self.noise_scale_max_value)

        image_prediction = noise_scale * torch.randn((batch_size, 3, image_size[1], image_size[0]), device=device, dtype=dtype)

        attention_mask_condition = {"full_attention": None}
        attention_mask_uncondition = {"full_attention": None}

        timesteps = torch.linspace(0.0, 1.0, num_steps+1, device=device)

        if enable_timestep_shift:
            timesteps = self._apply_time_schedule(timesteps, token_h*token_w, timestep_shift)

        for step_i in range(num_steps):
            t = timesteps[step_i]
            t_next = timesteps[step_i + 1]

            z = self.patchify(image_prediction, self.patch_size * merge_size)
            image_input = self.patchify(image_prediction, self.patch_size, channel_first=True)
            image_embeds = self.extract_feature(image_input.view(batch_size * grid_h*grid_w, -1), gen_model=True, grid_hw=grid_hw).view(batch_size, token_h*token_w, -1)
            t_expanded = t.expand(batch_size*token_h*token_w)
            timestep_embeddings = self.fm_modules['timestep_embedder'](t_expanded).view(batch_size, token_h*token_w, -1)
            if self.add_noise_scale_embedding:
                noise_scale_tensor = torch.full_like(t_expanded, noise_scale / self.noise_scale_max_value)
                noise_embeddings = self.fm_modules['noise_scale_embedder'](noise_scale_tensor).view(batch_size, token_h*token_w, -1)
                timestep_embeddings += noise_embeddings
            image_embeds = image_embeds + timestep_embeddings


            v_pred_condition = self._t2i_predict_v(image_embeds, indexes_image_condition, attention_mask_condition, past_key_values_condition, t, z, image_token_num=token_h*token_w,
                                                   timestep_embeddings=timestep_embeddings, image_size=image_size)


            if t > cfg_interval[0] and t < cfg_interval[1] and cfg_scale > 1:
                v_pred_uncondition = self._t2i_predict_v(image_embeds, indexes_image_uncondition, attention_mask_uncondition, past_key_values_uncondition, t, z, image_token_num=token_h*token_w,
                                                         timestep_embeddings=timestep_embeddings, image_size=image_size)
                if cfg_norm == 'cfg_zero_star':
                    positive_flat = v_pred_condition.view(batch_size, -1)
                    negative_flat = v_pred_uncondition.view(batch_size, -1)

                    alpha = optimized_scale(positive_flat,negative_flat)
                    alpha = alpha.view(batch_size, *([1] * (len(v_pred_condition.shape) - 1)))
                    alpha = alpha.to(positive_flat.dtype)

                    if (step_i <= 0):
                        v_pred = v_pred_condition*0.
                    else:
                        v_pred = v_pred_uncondition * alpha + cfg_scale * (v_pred_condition - v_pred_uncondition * alpha)
                else:
                    v_pred = v_pred_uncondition + cfg_scale * (v_pred_condition - v_pred_uncondition)
                    if cfg_norm == 'global':
                        norm_v_condition = torch.norm(v_pred_condition, dim=(1,2), keepdim=True)
                        norm_v_cfg = torch.norm(v_pred, dim=(1,2), keepdim=True)
                        scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                        v_pred = v_pred * scale
            else:
                v_pred = v_pred_condition

            z = z + (t_next - t) * v_pred

            image_prediction = self.unpatchify(z, self.patch_size * merge_size, image_size[1], image_size[0])

        clear_flash_kv_cache(past_key_values_condition)
        clear_flash_kv_cache(past_key_values_uncondition)

        return image_prediction


    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, value):
        return self.language_model.set_output_embeddings(value)

    def get_thw_indexes(self, input_ids, grid_hw=None):
        img_start_shift = torch.cat([torch.zeros(1, dtype=torch.long).to(input_ids.device),
                                     (input_ids == self.img_start_token_id).long()], dim=0)[:-1]
        not_img_token = (input_ids != self.img_context_token_id).long()
        t_indexes = ((img_start_shift + not_img_token).cumsum(0) - 1)
        h_indexes = torch.zeros_like(t_indexes).to(t_indexes.device)
        w_indexes = torch.zeros_like(t_indexes).to(t_indexes.device)

        if grid_hw is not None:
            selected = (input_ids == self.img_context_token_id)
            if selected.long().sum() > 0:
                abs_pos_w, abs_pos_h = build_abs_positions_from_grid_hw(
                    grid_hw // int(1 / self.downsample_ratio), device=t_indexes.device)
                h_indexes[selected] = abs_pos_h.to(t_indexes.device, t_indexes.dtype)
                w_indexes[selected] = abs_pos_w.to(t_indexes.device, t_indexes.dtype)
        return torch.stack([t_indexes, h_indexes, w_indexes], dim=0)


NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD  = [0.5, 0.5, 0.5]

class NEOX2I:
    def __init__(self, model_path, device):
        self.device = device
        self.model: NEOChatModel = NEOChatModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)

        self.model.eval()

    def _denorm(self, x: torch.Tensor, mean=NORM_MEAN, std=NORM_STD):
        """
        x: [B,3,H,W] normalized ((img-mean)/std). returns [0,1] clamped.
        """
        mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std  = torch.tensor(std,  device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x * std + mean).clamp(0, 1)

    def _get_dynamic_cache(self, past_kv):
        """
        past_kv ( L, 2, H // 2, P * S, D)
        """
        past_kv_dc = DynamicCache(config=self.model.language_model.model.config)
        L, _, H, S, D = past_kv.shape
        for layer_idx in range(L):
            k = past_kv[layer_idx][0].unsqueeze(0).to(self.device, non_blocking=True)
            v = past_kv[layer_idx][1].unsqueeze(0).to(self.device, non_blocking=True)
            past_kv_dc.update(key_states=k, value_states=v, layer_idx=layer_idx,)
        return past_kv_dc


    def t2i(self, past_kv, past_kv_txt, param: X2IParams):
        past_kv_dc = self._get_dynamic_cache(past_kv)
        past_kv_txt_dc = self._get_dynamic_cache(past_kv_txt)
        text_lens = (param.past_kvcache.get_compressed_len(),
                     param.past_kvcache_text.get_compressed_len())
        output = self.model.t2i_generate(
            past_key_values_condition=past_kv_dc,
            past_key_values_uncondition=past_kv_txt_dc,
            text_lens=text_lens,
            cfg_norm=param.get_cfg_norm(),
            cfg_scale=param.guidance_scale,
            image_size=(param.width, param.height),
            num_steps=param.steps,
            batch_size=param.num_images)

        return self._post_process(output)

    def _post_process(self, output):
        images = self._denorm(output)
        images = (images.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu()

        base64_images = [
            base64.b64encode(io.encode_jpeg(img).numpy()).decode("utf-8")
            for img in images
        ]
        return base64_images

    def it2i(self, past_kv, past_kv_txt, past_kv_img, param: X2IParams):
        past_kv_dc = self._get_dynamic_cache(past_kv)
        past_kv_txt_dc = self._get_dynamic_cache(past_kv_txt)
        past_kv_img_dc = self._get_dynamic_cache(past_kv_img)
        text_lens = (param.past_kvcache.get_compressed_len(),
                     param.past_kvcache_text.get_compressed_len(),
                     param.past_kvcache_img.get_compressed_len())
        output = self.model.it2i_generate(
            past_key_values_condition=past_kv_dc,
            past_key_values_text_uncondition=past_kv_txt_dc,
            past_key_values_img_uncondition=past_kv_img_dc,
            text_lens=text_lens,
            cfg_norm=param.get_cfg_norm(),
            cfg_scale=param.guidance_scale,
            img_cfg_scale=param.image_guidance_scale,
            image_size=(param.width, param.height),
            num_steps=param.steps,
            batch_size=param.num_images,
        )

        return self._post_process(output)
