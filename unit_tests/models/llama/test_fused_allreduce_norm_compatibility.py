import pytest

from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.stablelm.layer_infer.transformer_layer_infer import StablelmTransformerLayerInfer
from lightllm.models.starcoder2.layer_infer.transformer_layer_infer import Starcoder2TransformerLayerInfer


@pytest.mark.parametrize(
    "layer_cls",
    [
        StablelmTransformerLayerInfer,
        Starcoder2TransformerLayerInfer,
    ],
)
def test_layernorm_descendants_disable_fused_rmsnorm(monkeypatch, layer_cls):
    def init_llama(self, layer_num, network_config):
        self.network_config_ = network_config
        self._enable_fused_ar_add_norm = True

    monkeypatch.setattr(LlamaTransformerLayerInfer, "__init__", init_llama)

    layer = layer_cls(0, {})

    assert not layer._enable_fused_ar_add_norm
