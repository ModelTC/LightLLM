from lightllm.common.execution.formats.kv.spec import SparseMLAKVStateSpec
from lightllm.models.deepseek2.adapter import Deepseek2ModelAdapter
from dataclasses import asdict, replace
from lightllm.common.execution.compute.attention.sparse_mla.spec import SparseMLAAttentionSpec
from lightllm.models.deepseek3_2.layer_weights.transformer_layer_weight import Deepseek3_2TransformerLayerWeight
from lightllm.models.deepseek3_2.layer_infer.transformer_layer_infer import Deepseek3_2TransformerLayerInfer


class Deepseek3_2ModelAdapter(Deepseek2ModelAdapter):
    transformer_weight_class = Deepseek3_2TransformerLayerWeight
    transformer_layer_infer_class = Deepseek3_2TransformerLayerInfer

    def __init__(self, config, *, role, dtype, tp_world_size=1):
        super().__init__(config, role=role, dtype=dtype, tp_world_size=tp_world_size)
        quant = config.get("quantization_config") or {}
        if config["index_n_heads"] % tp_world_size or config.get("indexer_dim", 128) != config["index_head_dim"]:
            raise ValueError("indexer storage and TP projection dimensions must agree")
        self._attention = {
            layer: SparseMLAAttentionSpec(
                **asdict(spec),
                index_heads=config["index_n_heads"],
                index_dim=config["index_head_dim"],
                index_topk=config["index_topk"],
                index_quant_block=quant.get("weight_block_size", [128])[0],
                index_scale_fmt=quant.get("scale_fmt"),
            )
            for layer, spec in self._attention.items()
        }
        self._states = tuple(
            replace(
                state,
                params=SparseMLAKVStateSpec(
                    latent_dim=state.params.latent_dim,
                    rope_dim=state.params.rope_dim,
                    index_dim=config["index_head_dim"],
                ),
            )
            for state in self._states
        )

    def layer_attention_params(self, spec, weight, device):
        from lightllm.models.base.attention_params import sparse_mla_params

        return sparse_mla_params(spec, weight, device)
