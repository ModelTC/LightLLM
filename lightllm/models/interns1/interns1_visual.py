import os
import json
import torch
from PIL import Image
from typing import List, Optional, Union, Callable
from lightllm.server.multimodal_params import ImageItem
from lightllm.server.embed_cache.utils import read_shm, get_shm_name_data
from io import BytesIO
from lightllm.models.vit import get_load_image_func
from lightllm.utils.log_utils import init_logger
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoConfig
import torch.nn as nn
from transformers.activations import ACT2FN
import collections.abc
from transformers.utils import can_return_tuple, torch_int
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import BaseModelOutput
from lightllm.models.vit.triton_kernel.rms_norm_vit import rms_norm
from transformers.models.got_ocr2.image_processing_got_ocr2_fast import (
    GotOcr2ImageProcessorFast)
from transformers.image_utils import make_flat_list_of_images

logger = init_logger(__name__)

class InternS1VisionConfig(PretrainedConfig):
    model_type = "interns1_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        attention_bias=False,
        use_qk_norm=False,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        projection_dropout=0.0,
        initializer_range=0.02,
        norm_type="layer_norm",
        layer_norm_eps=1e-06,
        image_size=[448, 448],
        patch_size=[14, 14],
        num_channels=3,
        use_mask_token=False,
        use_absolute_position_embeddings=True,
        layer_scale_init_value=0.1,
        use_mean_pooling=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_bias = attention_bias
        self.use_qk_norm = use_qk_norm
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout = attention_dropout
        self.projection_dropout = projection_dropout
        self.initializer_range = initializer_range
        self.norm_type = norm_type
        self.layer_norm_eps = layer_norm_eps

        image_size = image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, (list, tuple)) else (patch_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size

        self.num_channels = num_channels
        self.use_mask_token = use_mask_token
        self.use_absolute_position_embeddings = use_absolute_position_embeddings
        self.layer_scale_init_value = layer_scale_init_value
        self.use_mean_pooling = use_mean_pooling

class  InternS1Config(PretrainedConfig):

    model_type = "interns1"
    sub_configs = {"text_config": AutoConfig, "vision_config": InternS1VisionConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=151667,
        image_seq_length=256,
        downsample_ratio=0.5,
        projector_hidden_act="gelu",
        vision_feature_layer=-1,
        vision_feature_select_strategy="default",
        **kwargs,
    ):
        from transformers import CONFIG_MAPPING

        self.image_token_id = image_token_id
        self.image_seq_length = image_seq_length
        self.downsample_ratio = downsample_ratio
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy

        if isinstance(vision_config, dict):
            self.vision_config = InternS1VisionConfig(**vision_config)
        elif isinstance(vision_config, InternS1VisionConfig):
            self.vision_config = vision_config
        elif vision_config is None:
            self.vision_config = InternS1VisionConfig()

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "qwen2"  # todo
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen2"]()  # todo

        self.text_config = text_config

        super().__init__(**kwargs)

class InternS1VisionPatchEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_shape = patch_shape

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        embeddings = self.projection(pixel_values.to(self.projection.weight.dtype))
        patch_height, patch_width = embeddings.shape[2], embeddings.shape[3]
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings, (patch_height, patch_width)

class InternS1VisionEmbeddings(nn.Module):
    def __init__(self, config: InternS1VisionConfig) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        if config.use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        else:
            self.mask_token = None
        self.patch_embeddings = InternS1VisionPatchEmbeddings(config)
        self.patch_size = config.patch_size
        self.image_size = (
            config.image_size
            if isinstance(config.image_size, collections.abc.Iterable)
            else (config.image_size, config.image_size)
        )
        num_patches = self.patch_embeddings.num_patches
        if config.use_absolute_position_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        else:
            self.position_embeddings = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size[0]
        new_width = width // self.patch_size[1]

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        embeddings, (patch_height, patch_width) = self.patch_embeddings(pixel_values)
        batch_size, seq_len, _ = embeddings.size()

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1 - w) + mask_tokens * w

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        if self.position_embeddings is not None:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        embeddings = self.dropout(embeddings)

        return embeddings, (patch_height, patch_width)

# 替换了算子
class InternS1VisionRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        InternS1VisionRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_norm(hidden_states, self.weight, eps=self.variance_epsilon)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = key
    value_states = value

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # No upcasting of the attention weights to float32 in this implementation
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

# TODO 验证完后改flashattention
class InternS1VisionAttention(nn.Module):
    """Attention Class for InternS1 Vision Encoder"""

    def __init__(self, config: InternS1VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        proj_dropout = config.projection_dropout
        qk_norm = config.use_qk_norm

        # Needed for flash attention
        self.is_causal = False

        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.projection_layer = nn.Linear(self.embed_dim, self.embed_dim)
        self.projection_dropout = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()

        self.q_norm = InternS1VisionRMSNorm(self.embed_dim) if qk_norm else nn.Identity()
        self.k_norm = InternS1VisionRMSNorm(self.embed_dim) if qk_norm else nn.Identity()


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        batch_size, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scale,
            is_causal=False,
            **kwargs,
        )
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        output = self.projection_layer(attn_output)
        output = self.projection_dropout(output)

        outputs = (output, attn_weights) if output_attentions else (output, None)
        return outputs


class InternS1VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


NORM2FN = {"layer_norm": nn.LayerNorm, "rms_norm": InternS1VisionRMSNorm}

class InternS1VisionLayer(GradientCheckpointingLayer):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: InternS1VisionConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = InternS1VisionAttention(config)
        self.mlp = InternS1VisionMLP(config)
        # InternS1 uses different layernorm implementations for different models
        self.layernorm_before = NORM2FN[config.norm_type](config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = NORM2FN[config.norm_type](config.hidden_size, eps=config.layer_norm_eps)

        init_values = config.layer_scale_init_value
        self.lambda_1 = nn.Parameter(init_values * torch.ones(config.hidden_size), requires_grad=True)
        self.lambda_2 = nn.Parameter(init_values * torch.ones(config.hidden_size), requires_grad=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> Union[tuple[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        attention_output, attention_weights = self.attention(
            self.layernorm_before(hidden_states),  # in InternS1Vision, layernorm is applied before self-attention
            output_attentions=output_attentions,
        )

        attention_output = self.lambda_1 * attention_output

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in InternS1Vision, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        layer_output = self.mlp(layer_output)
        layer_output = self.dropout(layer_output)

        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output

        # second residual connection
        layer_output = layer_output + hidden_states

        return layer_output, attention_weights


class InternS1VisionEncoder(nn.Module):
    def __init__(self, config: InternS1VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([InternS1VisionLayer(config) for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class InternS1VisionModel(PreTrainedModel):
    def __init__(self, config: InternS1VisionConfig) -> None:
        super().__init__(config)
        self.config = config

        self.embeddings = InternS1VisionEmbeddings(config)
        self.encoder = InternS1VisionEncoder(config)

        self.layernorm = (
            nn.Identity() if config.use_mean_pooling else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings


    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        embedding_output, _ = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class InternS1MultiModalProjector(nn.Module):
    def __init__(self, config: InternS1Config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.vision_config.hidden_size * int(1 / config.downsample_ratio) ** 2,dtype=config.torch_dtype)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * int(1 / config.downsample_ratio) ** 2, config.text_config.hidden_size,dtype=config.torch_dtype
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size,dtype=config.torch_dtype)

    def forward(self, image_features):
        hidden_states = self.layer_norm(image_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

class InternS1VisionTransformer:
    def __init__(self):
        pass

    def load_model(self, weight_dir):
        assert torch.cuda.is_available()
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.config = json.load(open(os.path.join(weight_dir, "config.json")))

        cfg = InternS1Config(**self.config)
        self.vision_tower = InternS1VisionModel._from_config(config=cfg.vision_config,torch_dtype=self.dtype)
        self.multi_modal_projector = InternS1MultiModalProjector(cfg)

        self.vision_tower.eval().cuda()
        self.multi_modal_projector.eval().cuda()

        self.image_processor = GotOcr2ImageProcessorFast.from_pretrained(weight_dir)


    
    def cuda(self):
        return self

    def pixel_shuffle(self, vision_features: torch.Tensor, scale_factor: float = 0.5):
        batch_size, width, height, channels = vision_features.size()

        if height % scale_factor != 0 or width % scale_factor != 0:
            raise ValueError("Height and width must be divisible by scale_factor for proper downsampling.")

        # Reshape to allow downsampling
        vision_features = vision_features.view(
            batch_size, width, int(height * scale_factor), int(channels / scale_factor)
        )
        # Permute dimensions to align downsampled axis correctly
        vision_features = vision_features.permute(0, 2, 1, 3).contiguous()

        # Reshape to achieve final downsampled dimensions
        vision_features = vision_features.view(
            batch_size, int(height * scale_factor), int(width * scale_factor), int(channels / (scale_factor**2))
        )

        # Swap height and width back for proper orientation
        vision_features = vision_features.permute(0, 2, 1, 3).contiguous()

        return vision_features


    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:

        vision_feature_layer = self.config['vision_feature_layer']
        vision_feature_select_strategy = self.config['vision_feature_select_strategy']

        downsample_ratio = self.config['downsample_ratio']
        if vision_feature_layer == -1:
            vision_features = self.vision_tower(pixel_values=pixel_values).last_hidden_state
        else:
            vision_features = self.vision_model(pixel_values=pixel_values).hidden_states[vision_feature_layer]
        if vision_feature_select_strategy == "default":
            vision_features = vision_features[:, 1:, :]

        # Calculate dimensions based on vision features
        channels = vision_features.shape[1]
        feature_size = int(channels**0.5)
        batch_size = vision_features.shape[0]

        # Reshape tensor to spatial dimensions
        vision_features = vision_features.reshape(batch_size, feature_size, feature_size, -1)

        # Apply downsampling using pixel shuffle
        vision_features = self.pixel_shuffle(vision_features, scale_factor=downsample_ratio)

        # Reshape tensor to prepare for projection
        vision_features = vision_features.reshape(batch_size, -1, vision_features.shape[-1])

        # Project features through multi-modal projector
        vision_features = self.multi_modal_projector(vision_features)
        return vision_features

    def encode(self, images: List[ImageItem]):
        img_tensors = []
        valid_ids = []
        valid_id = 0
        uuids = []

        for i, img in enumerate(images):
            if isinstance(img, ImageItem):
                uuids.append(img.uuid)
                image_data = read_shm(get_shm_name_data(img.uuid))
                image_data = Image.open(BytesIO(image_data))

                images = make_flat_list_of_images(image_data)
                processor_args = self.image_processor.to_dict()
                processor_args.pop("_processor_class", None)
                processor_args.pop("image_processor_type", None)
                processor_args["return_tensors"] = "pt"
                image_inputs = self.image_processor(images=images, **processor_args)
                image_pixel_values = image_inputs.pop("pixel_values")
                img_tensors.append(image_pixel_values)

            else:
                raise Exception("Unsupport input types: {} for {}".format(type(img), img))

            cur_num = img_tensors[-1].shape[0]
            valid_ids.append([valid_id, valid_id + cur_num])
            valid_id += cur_num

        if len(img_tensors) <= 0:
            return None

        imgs = torch.cat(img_tensors, dim=0)
        pixel_values = imgs.cuda().to(dtype=self.dtype, non_blocking=True)
        all_img_embeds = self.extract_feature(pixel_values)

        return all_img_embeds, uuids, valid_ids



