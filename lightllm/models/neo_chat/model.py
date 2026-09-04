import os
import json
import torch
from lightllm.common.build_utils import repair_config
from lightllm.models.registry import ModelRegistry, llm_model_type_is
from lightllm.models.qwen3_vl.infer_struct import Qwen3VLInferStateInfo
from lightllm.models.qwen3_vl.layer_infer.pre_layer_infer import Qwen3VLMultimodalPreLayerInfer
from lightllm.models.qwen3_vl.layer_infer.transformer_layer_infer import Qwen3VLTransformerLayerInfer
from lightllm.models.qwen3_vl.layer_weights.pre_and_post_layer_weight import Qwen3VLPreAndPostLayerWeight
from lightllm.models.qwen2_vl.model import QWen2VLTokenizer
from lightllm.models.qwen3.model import Qwen3TpPartModel
from lightllm.server.core.objs import SamplingParams
from lightllm.models.qwen3_moe.model import Qwen3MOEModel
from lightllm.server.multimodal_params import AudioItem, MultimodalParams, ImageItem
from lightllm.models.neo_chat_moe.vision_process import smart_resize
from lightllm.models.internvl.model import InternvlTokenizer
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.models.neo_chat.layer_infer.transformer_layer_infer import NeoChatTransformerLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.neo_chat.layer_weights.transformer_layer_weight import NeoChatTransformerLayerWeight
from lightllm.models.neo_chat.layer_weights.pre_and_post_layer_weight import NeoChatPreAndPostLayerWeight
from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.models.neo_chat_moe.infer_struct import NeoChatInferStateInfo
from lightllm.common.basemodel.attention import (
    get_neo_prefill_att_backend_class,
    get_neo_decode_att_backend_class,
    BaseAttBackend,
)
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


class NeoModelBase:
    def _init_custom(self):

        super()._init_custom()

        if "rope_theta_hw" in self.config:
            self._init_to_get_hw_rotary()

    def _init_to_get_rotary(self, default_base=10000):
        partial_head_dim = int(self.config.get("partial_rotary_factor", 1) * self.head_dim_)
        if self.config.get("rope_scaling", {}) is None:
            rope_scaling_factor = 1.0
        else:
            rope_scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)

        base = self.config.get("rope_theta", float(default_base))

        if "max_sequence_length" in self.config:
            max_seq_len = self.config["max_sequence_length"]
        else:
            max_position_embeddings = self.config.get(
                "max_position_embeddings", 2048 if base <= 10000.0 + 1e-5 else 16384
            )
            max_seq_len = max_position_embeddings * rope_scaling_factor

        # NTK
        try:
            ntk_alpha = float(os.environ.get("LIGHTLLM_NTK_ALPHA", 1))
            assert ntk_alpha >= 1
            if ntk_alpha > 1:
                logger.info(f"Note: NTK enabled, alpha set to {ntk_alpha}")
            max_seq_len *= ntk_alpha
            base = base * (ntk_alpha ** (partial_head_dim / (partial_head_dim - 2)))  # Base change formula
        except:
            pass

        full_inv_freq = 1.0 / (
            base ** (torch.arange(0, partial_head_dim, 2, device="cpu", dtype=torch.float32) / partial_head_dim)
        )
        inv_freq = full_inv_freq[::2]
        t = (
            torch.arange(max(max_seq_len + 1024 * 128, self.max_seq_length), device="cpu", dtype=torch.float32)
            / rope_scaling_factor
        )
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(self.data_type).cuda()
        self._sin_cached = torch.sin(freqs).to(self.data_type).cuda()
        return

    def _init_to_get_hw_rotary(self, default_base=10000):
        partial_head_dim = int(self.config.get("partial_rotary_factor", 1) * self.head_dim_ // 2)
        if self.config.get("rope_scaling", {}) is None:
            rope_scaling_factor = 1.0
        else:
            rope_scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)

        base = self.config.get("rope_theta_hw", float(default_base))
        if "max_sequence_length" in self.config:
            max_seq_len = self.config["max_sequence_length"]
        else:
            max_position_embeddings = self.config.get(
                "max_position_embeddings_hw", 2048 if base <= 10000.0 + 1e-5 else 16384
            )
            max_seq_len = max_position_embeddings * rope_scaling_factor

        # NTK
        try:
            ntk_alpha = float(os.environ.get("LIGHTLLM_NTK_ALPHA", 1))
            assert ntk_alpha >= 1
            if ntk_alpha > 1:
                logger.info(f"Note: NTK enabled, alpha set to {ntk_alpha}")
            max_seq_len *= ntk_alpha
            base = base * (ntk_alpha ** (partial_head_dim / (partial_head_dim - 2)))  # Base change formula
        except:
            pass
        full_inv_freq = 1.0 / (
            base ** (torch.arange(0, partial_head_dim, 2, device="cpu", dtype=torch.float32) / partial_head_dim)
        )
        inv_freq = full_inv_freq[::2]
        t = (
            torch.arange(max(max_seq_len + 1024 * 128, self.max_seq_length), device="cpu", dtype=torch.float32)
            / rope_scaling_factor
        )
        freqs = torch.outer(t, inv_freq)

        self._hw_cos_cached = torch.cos(freqs).to(self.data_type).cuda()
        self._hw_sin_cached = torch.sin(freqs).to(self.data_type).cuda()
        return


@ModelRegistry(["neo_chat"], is_multimodal=True, condition=llm_model_type_is("qwen3"))
class NeoTpPartModel(NeoModelBase, Qwen3TpPartModel):

    pre_layer_infer_class = LlamaMultimodalPreLayerInfer
    transformer_layer_infer_class = NeoChatTransformerLayerInfer

    pre_and_post_weight_class = NeoChatPreAndPostLayerWeight
    transformer_weight_class = NeoChatTransformerLayerWeight

    infer_state_class = NeoChatInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_inferstate_cls(self):
        pass

    def _init_att_backend(self):
        self.prefill_att_backend: BaseAttBackend = get_neo_prefill_att_backend_class(
            index=0, priority_list=["fa3", "triton"]
        )(model=self)
        self.decode_att_backend: BaseAttBackend = get_neo_decode_att_backend_class(
            index=0, priority_list=["fa3", "triton"]
        )(model=self)

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            all_config = json.load(json_file)
            self.config = all_config["llm_config"]
        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size
        return
