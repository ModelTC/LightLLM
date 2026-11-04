from lightllm.models.qwen3_vl.adapter import Qwen3VLModelAdapter
from lightllm.common.basemodel import TpPartBaseModel
from lightllm.models.qwen3_vl.infer_struct import Qwen3VLInferStateInfo
from lightllm.models.qwen3_vl.layer_infer.pre_layer_infer import Qwen3VLMultimodalPreLayerInfer
from lightllm.models.qwen3_vl.layer_infer.transformer_layer_infer import Qwen3VLTransformerLayerInfer
from lightllm.models.qwen3_vl.layer_weights.pre_and_post_layer_weight import Qwen3VLPreAndPostLayerWeight
from lightllm.models.qwen2_vl.model import QWen2VLTokenizer
from lightllm.models.qwen3.model import Qwen3TpPartModel
from lightllm.models.qwen3.adapter import Qwen3ModelAdapter


class QWen3VLTokenizer(QWen2VLTokenizer):
    def __init__(self, tokenizer=None, image_processor=None, **kwargs):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.min_pixel = self.image_processor.size["shortest_edge"]
        self.max_pixel = self.image_processor.size["longest_edge"]
        self.patch_size = self.image_processor.patch_size
        self.merge_size = self.image_processor.merge_size
        self.image_start_id = kwargs["model_cfg"]["vision_start_token_id"]
        self.image_end_id = kwargs["model_cfg"]["vision_end_token_id"]
        self.image_token_id = kwargs["model_cfg"]["image_token_id"]


class Qwen3VLTpPartModel(Qwen3TpPartModel):
    model_adapter_class = Qwen3VLModelAdapter
    pre_layer_infer_class = Qwen3VLModelAdapter.pre_layer_infer_class
    transformer_layer_infer_class = Qwen3VLModelAdapter.transformer_layer_infer_class
    pre_and_post_weight_class = Qwen3VLModelAdapter.pre_and_post_weight_class
    infer_state_class = Qwen3VLInferStateInfo

    def _init_config(self):
        TpPartBaseModel._init_config(self)
