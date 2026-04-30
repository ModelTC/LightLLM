import os
import json
import torch
from transformers import AutoConfig
from lightllm.models.registry import ModelRegistry
from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.common.basemodel.attention.triton.fp import TritonAttBackend
from lightllm.common.kv_cache_mem_manager.mem_utils import select_mem_manager_class
from lightllm.common.build_utils import repair_config
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.gemma4.infer_struct import Gemma4InferStateInfo
from lightllm.models.gemma4.layer_infer.pre_layer_infer import Gemma4PreLayerInfer
from lightllm.models.gemma4.layer_infer.post_layer_infer import Gemma4PostLayerInfer
from lightllm.models.gemma4.layer_infer.transformer_layer_infer import Gemma4TransformerLayerInfer
from lightllm.models.gemma4.layer_weights.pre_and_post_layer_weight import Gemma4PreAndPostLayerWeight
from lightllm.models.gemma4.layer_weights.transformer_layer_weight import Gemma4TransformerLayerWeight
from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num, get_env_start_args
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class Gemma4Tokenizer(BaseMultiModalTokenizer):
    """
    Thin wrapper; Phase-A milestone only exercises the text path. Multimodal
    splice logic will be added alongside the vision tower port (Phase B).
    """

    def __init__(self, tokenizer, model_cfg):
        super().__init__(tokenizer)
        self.image_token_index = model_cfg.get("image_token_id", 258880)
        self.boi_token_index = model_cfg.get("boi_token_id", 255999)
        self.eoi_token_index = model_cfg.get("eoi_token_id", 258882)
        self.image_length = model_cfg.get("vision_soft_tokens_per_image", 280)
        # Gemma-4's tokenizer ships with `add_bos_token=False`, and even
        # `add_special_tokens=True` doesn't prepend `<bos>`. The model generates
        # garbage without it, so we always prepend manually.
        self.bos_token_id = tokenizer.bos_token_id

    def init_imageitem_extral_params(self, img, multi_params, sampling_params):
        return

    def init_audioitem_extral_params(self, audio, multi_params, sampling_params):
        raise NotImplementedError

    def get_image_token_length(self, img):
        return self.image_length

    def get_audio_token_length(self, audio):
        raise NotImplementedError

    def encode(self, prompt, multimodal_params=None, add_special_tokens=False):
        # Text-only path for Phase A — reject image/audio input loudly so users
        # know multimodal isn't wired yet.
        if multimodal_params is not None and (
            getattr(multimodal_params, "images", None) or getattr(multimodal_params, "audios", None)
        ):
            raise NotImplementedError(
                "Gemma-4 multimodal (image/audio) inference is not yet implemented in LightLLM; "
                "only text prompts are supported for now."
            )
        input_ids = self.tokenizer(prompt).input_ids
        # Auto-prepend <bos> for prompts (Gemma-4 generates garbage without it),
        # but honour `add_special_tokens=False` so callers like stop-sequence
        # encoding can opt out — otherwise stop strings get a leading BOS that
        # never appears in generated output and never matches.
        if (
            add_special_tokens
            and self.bos_token_id is not None
            and (len(input_ids) == 0 or input_ids[0] != self.bos_token_id)
        ):
            input_ids = [self.bos_token_id] + input_ids
        return input_ids


@ModelRegistry("gemma4", is_multimodal=True)
class Gemma4TpPartModel(LlamaTpPartModel):
    pre_and_post_weight_class = Gemma4PreAndPostLayerWeight
    transformer_weight_class = Gemma4TransformerLayerWeight

    pre_layer_infer_class = Gemma4PreLayerInfer
    transformer_layer_infer_class = Gemma4TransformerLayerInfer
    post_layer_infer_class = Gemma4PostLayerInfer

    infer_state_class = Gemma4InferStateInfo

    def __init__(self, kvargs):
        # head_dim_ is used by the default _init_to_get_rotary which we
        # override; still set it to the sliding-layer head_dim for consistency
        # with the mem manager and any generic helpers.
        self.head_dim_ = 256
        super().__init__(kvargs)
        return

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            self.config = json.load(json_file)
        # The shipped checkpoint is a multimodal config wrapping a Gemma4TextConfig
        # under text_config; flatten it so downstream code sees text-model fields
        # at the top level (mirrors the gemma3 approach).
        if "text_config" in self.config:
            hf_config = AutoConfig.from_pretrained(self.weight_dir_, trust_remote_code=True)
            self.config = hf_config.text_config.to_dict()

        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        return

    def _verify_params(self):
        assert self.load_way == "HF", "Gemma-4 only supports HF format."
        assert self.config["num_attention_heads"] % self.tp_world_size_ == 0
        assert self.config["num_key_value_heads"] % self.tp_world_size_ == 0
        num_global_kv = self.config.get("num_global_key_value_heads", self.config["num_key_value_heads"])
        assert num_global_kv % self.tp_world_size_ == 0, (
            f"num_global_key_value_heads={num_global_kv} must be divisible by tp={self.tp_world_size_}"
        )
        return

    def _init_mem_manager(self):
        # Uniform per-layer KV cache layout keyed to the *sliding* attention shape
        # (num_kv_heads=16, head_dim=256). Full-attention layers (num_kv_heads=4,
        # head_dim=512, k_eq_v) reuse the same byte budget at <=50% utilization;
        # the transformer-layer infer code handles the reshape when reading back.
        head_num_per_rank = self.config["num_key_value_heads"] // self.tp_world_size_
        head_dim = self.config["head_dim"]
        self.mem_manager = select_mem_manager_class()(
            self.max_total_token_num,
            dtype=self.data_type,
            head_num=head_num_per_rank,
            head_dim=head_dim,
            layer_num=self.config["num_hidden_layers"] + get_added_mtp_kv_layer_num(),
            mem_fraction=self.mem_fraction,
        )
        return

    def _init_att_backend(self):
        # Gemma-4 has per-layer heterogeneous attention shape (sliding layers
        # use head_dim=256/16 KV heads, full-attn layers use head_dim=512/4).
        # The flashinfer backend in this repo plans once per infer_state with
        # a single (head_dim, num_kv_heads), so it crashes / silently produces
        # wrong results on the layer where the shape doesn't match. FA3 reads
        # head_dim and num_kv_heads from the per-call tensor shapes, so it
        # supports the heterogeneous layout AND honours per-call sliding window
        # — which is what we want on sliding layers.
        from lightllm.common.basemodel.attention.fa3.fp import Fa3AttBackend
        from lightllm.utils.sgl_utils import flash_attn_with_kvcache

        fa3_loadable = flash_attn_with_kvcache is not None
        args = get_env_start_args()
        backends = set(args.llm_prefill_att_backend + args.llm_decode_att_backend)
        for backend_name in backends:
            assert backend_name in ("auto", "triton", "fa3"), (
                "Gemma-4 requires triton or fa3 (per-layer dynamic head_dim / "
                "num_kv_heads); flashinfer is not wired for the heterogeneous "
                f"layout. Got --llm_*_att_backend={backend_name!r}."
            )
        if "fa3" in backends:
            assert fa3_loadable, (
                "Requested --llm_*_att_backend=fa3 but neither sgl_kernel nor "
                "flash_attn_3 (flash_attn_interface) imported successfully. "
                "Build flash-attention/hopper from source against the current torch."
            )
        # Default policy: prefer FA3 if available (gets us real sliding-window
        # attention on sliding layers); fall back to triton otherwise.
        prefer_fa3 = fa3_loadable and (backends <= {"auto", "fa3"})
        if prefer_fa3:
            self.prefill_att_backend = Fa3AttBackend(model=self)
            self.decode_att_backend = Fa3AttBackend(model=self)
            self.config["_gemma4_use_swa"] = True
        else:
            self.prefill_att_backend = TritonAttBackend(model=self)
            self.decode_att_backend = TritonAttBackend(model=self)
            self.config["_gemma4_use_swa"] = False

    def _init_custom(self):
        self._init_to_get_rotary_gemma4()

    def _init_to_get_rotary_gemma4(self):
        rope_params = self.config["rope_parameters"]

        # Cap the rotary table at something we can fit in memory — Gemma-4's
        # advertised max_position_embeddings is 262144 which would require
        # ~200MB per table in fp32. Rely on the server's max_seq_length instead.
        max_seq_len = max(self.max_seq_length + 1024, 16384)

        t = torch.arange(max_seq_len, dtype=torch.float32, device="cpu")

        # Sliding layers: default RoPE, theta=10000, full rotation over head_dim=256.
        sliding_params = rope_params["sliding_attention"]
        sliding_head_dim = self.config["head_dim"]
        sliding_theta = sliding_params["rope_theta"]
        sliding_partial = sliding_params.get("partial_rotary_factor", 1.0)
        sliding_rot_dim = int(sliding_head_dim * sliding_partial)
        inv_freq_sliding = 1.0 / (
            sliding_theta ** (torch.arange(0, sliding_rot_dim, 2, dtype=torch.float32) / sliding_rot_dim)
        )
        freqs_s = torch.outer(t, inv_freq_sliding)
        self._cos_cached_sliding = torch.cos(freqs_s).to(self.data_type).cuda()
        self._sin_cached_sliding = torch.sin(freqs_s).to(self.data_type).cuda()

        # Full-attention layers: proportional RoPE, theta=1_000_000,
        # partial_rotary_factor=0.25 over global_head_dim=512.
        # Proportional semantics (HF transformers):
        #   rope_angles = int(partial * head_dim // 2)   -> 64
        #   inv_freq[0:rope_angles] = 1 / base ** (arange(0, 2*rope_angles, 2) / head_dim)
        #   inv_freq[rope_angles:head_dim//2] = 0   (identity rotation for "no-pe" dims)
        full_params = rope_params["full_attention"]
        full_head_dim = self.config["global_head_dim"]
        full_theta = full_params["rope_theta"]
        full_partial = full_params.get("partial_rotary_factor", 1.0)
        rope_type = full_params.get("rope_type", "default")
        if rope_type == "proportional":
            rope_angles = int(full_partial * full_head_dim // 2)
            inv_freq_rot = 1.0 / (
                full_theta
                ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / full_head_dim)
            )
            nope_angles = full_head_dim // 2 - rope_angles
            if nope_angles > 0:
                inv_freq_full = torch.cat(
                    [inv_freq_rot, torch.zeros(nope_angles, dtype=torch.float32)]
                )
            else:
                inv_freq_full = inv_freq_rot
        else:
            full_rot_dim = int(full_head_dim * full_partial)
            inv_freq_full = 1.0 / (
                full_theta ** (torch.arange(0, full_rot_dim, 2, dtype=torch.float32) / full_rot_dim)
            )

        freqs_f = torch.outer(t, inv_freq_full)
        self._cos_cached_full = torch.cos(freqs_f).to(self.data_type).cuda()
        self._sin_cached_full = torch.sin(freqs_f).to(self.data_type).cuda()
        return
