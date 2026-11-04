from lightllm.models.gpt_oss.layer_infer.transformer_layer_infer import GptOssTransformerLayerInfer
from lightllm.models.gpt_oss.layer_weights.transformer_layer_weight import GptOssTransformerLayerWeight
from dataclasses import replace

from lightllm.models.llama.model import LlamaTpPartModel, LlamaModelAdapter
from lightllm.common.execution.specs import MoeSpec
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.common.basemodel.attention import get_prefill_att_backend_class, get_decode_att_backend_class
from lightllm.common.basemodel.attention import BaseAttBackend

logger = init_logger(__name__)


class GptOssModelAdapter(LlamaModelAdapter):
    transformer_weight_class = GptOssTransformerLayerWeight
    transformer_layer_infer_class = GptOssTransformerLayerInfer

    def __init__(self, config, **kwargs):
        config = dict(config)
        config.setdefault(
            "layer_types",
            ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(config["num_hidden_layers"])],
        )
        super().__init__(config, **kwargs)
        self._attention = {i: replace(spec, attention_sink=True) for i, spec in self._attention.items()}

    def moe_specs(self):
        config = self.config
        spec = MoeSpec(
            config["hidden_size"],
            config["intermediate_size"],
            config["num_local_experts"],
            config["num_experts_per_tok"],
            self.dtype,
            renormalize=True,
            scoring="topk_softmax",
            alpha=1.702,
            limit=7.0,
            round_intermediates=True,
        )
        return {layer: spec for layer in self.layer_ids}


class GptOssTpPartModel(LlamaTpPartModel):
    model_adapter_class = GptOssModelAdapter
    # weight class
    transformer_weight_class = GptOssTransformerLayerWeight

    # infer class
    transformer_layer_infer_class = GptOssTransformerLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)

    def _init_att_backend(self):
        self.prefill_att_backend: BaseAttBackend = get_prefill_att_backend_class(index=0, priority_list=["fa3"])(
            model=self
        )
        self.decode_att_backend: BaseAttBackend = get_decode_att_backend_class(index=0, priority_list=["fa3"])(
            model=self
        )
