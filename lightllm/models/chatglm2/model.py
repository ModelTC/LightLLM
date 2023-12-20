import os
import json
import torch

from lightllm.models.chatglm2.layer_infer.transformer_layer_infer import ChatGLM2TransformerLayerInfer
from lightllm.models.chatglm2.layer_weights.transformer_layer_weight import ChatGLM2TransformerLayerWeight
from lightllm.models.chatglm2.layer_weights.pre_and_post_layer_weight import ChatGLM2PreAndPostLayerWeight
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.common.build_utils import repair_config
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

class ChatGlm2TpPartModel(LlamaTpPartModel):
    # Please use the fast tokenizer from:
    # [THUDM/chatglm3-6b PR #12](https://huggingface.co/THUDM/chatglm3-6b/discussions/12).

    # weight class
    pre_and_post_weight_class = ChatGLM2PreAndPostLayerWeight
    transformer_weight_class = ChatGLM2TransformerLayerWeight

    # infer class
    transformer_layer_infer_class = ChatGLM2TransformerLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
    
    def _init_config(self):
        super()._init_config()
        # rename key
        # repair_config()
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer", "num_layers"])
        repair_config(self.config, same_names=["vocab_size", "padded_vocab_size"])
        repair_config(self.config, same_names=["rms_norm_eps", "layernorm_epsilon"])
        repair_config(self.config, same_names=["seq_length", "max_sequence_length"])
        return
    
    def _reset_num_key_value_heads(self):
        self.config["num_key_value_heads"] = self.config["multi_query_group_num"]
        return
    
    def _verify_params(self):
        assert self.load_way == "HF", "ChatGLM only support HF format for now"
        assert self.world_size_ in [1, 2], "ChatGLM can only run in tp=1 or tp=2 for now"

    def _init_to_get_rotary(self, base=10000):
        if self.config.get("rope_scaling", {}) is None:
            rope_scaling_factor = 1.0
        else:
            rope_scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)
        if "max_sequence_length" in self.config:
            max_seq_len = self.config["max_sequence_length"]
        else:
            max_seq_len = self.config.get("max_position_embeddings", 2048) * rope_scaling_factor

        base = float(base) * self.config.get("rope_ratio", 1.0)

        # NTK
        try:
            ntk_alpha = float(os.environ.get("LIGHTLLM_NTK_ALPHA", 1))
            assert ntk_alpha >= 1
            if ntk_alpha > 1:
                logger.info(f"Note: NTK enabled, alpha set to {ntk_alpha}")
            max_seq_len *= ntk_alpha
            base = base * (ntk_alpha ** (self.head_dim_ / (self.head_dim_-2))) #Base change formula
        except:
            pass
        n_elem = self.head_dim_ // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, n_elem, 2, device="cpu", dtype=torch.float32) / n_elem))
        t = torch.arange(max_seq_len + 1024 * 64, device="cpu", dtype=torch.float32) / rope_scaling_factor
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(torch.float16).cuda()
        self._sin_cached = torch.sin(freqs).to(torch.float16).cuda()
        return
