import os
import torch
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.layer_infer.template.transformer_layer_infer_cohere_template import (
    TransformerLayerCohereInferTpl,
)
from lightllm.common.kv_cache_mem_manager import MemoryManager
from lightllm.models.registry import ModelRegistry
from lightllm.models.cohere.infer_struct import CohereInferStateInfo
from lightllm.models.cohere.layer_infer.post_layer_infer import CoherePostLayerInfer
from lightllm.models.cohere.layer_infer.transformer_layer_infer import CohereTransformerLayerInfer
from lightllm.models.cohere.layer_weights.pre_and_post_layer_weight import CoherePreAndPostLayerWeight
from lightllm.models.cohere.layer_weights.transformer_layer_weight import CohereTransformerLayerWeight
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@ModelRegistry("cohere")
class CohereTpPartModel(LlamaTpPartModel):
    pre_and_post_weight_class = CoherePreAndPostLayerWeight
    transformer_weight_class = CohereTransformerLayerWeight

    pre_layer_infer_class = LlamaPreLayerInfer
    transformer_layer_infer_class = CohereTransformerLayerInfer
    post_layer_infer_class = CoherePostLayerInfer

    infer_state_class = CohereInferStateInfo

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

        inv_freq = 1.0 / (
            base ** (torch.arange(0, partial_head_dim, 2, device="cpu", dtype=torch.float32) / partial_head_dim)
        )
        t = torch.arange(max_seq_len + 1024 * 128, device="cpu", dtype=torch.float32) / rope_scaling_factor
        freqs = torch.outer(t, inv_freq)
        freqs = torch.repeat_interleave(freqs, 2, dim=-1)

        self._cos_cached = torch.cos(freqs).to(self.data_type).cuda()
        self._sin_cached = torch.sin(freqs).to(self.data_type).cuda()
        return
