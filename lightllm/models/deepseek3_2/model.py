from lightllm.models.deepseek2.model import Deepseek2TpPartModel
from lightllm.models.deepseek3_2.layer_weights.transformer_layer_weight import Deepseek3_2TransformerLayerWeight
from lightllm.models.deepseek3_2.layer_infer.transformer_layer_infer import Deepseek3_2TransformerLayerInfer
from lightllm.common.basemodel.attention import get_nsa_prefill_att_backend_class, get_nsa_decode_att_backend_class


class Deepseek3_2TpPartModel(Deepseek2TpPartModel):

    # weight class
    transformer_weight_class = Deepseek3_2TransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Deepseek3_2TransformerLayerInfer

    @classmethod
    def _full_attention_options(cls, config, args):
        options = super()._full_attention_options(config, args)
        options["packed"] = True
        return options

    def _init_att_backend(self):
        self.prefill_att_backend = get_nsa_prefill_att_backend_class(index=0)(model=self)
        self.decode_att_backend = get_nsa_decode_att_backend_class(index=0)(model=self)
        return
