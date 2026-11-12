from lightllm.common.basemodel.triton_kernel.mhc import hc_combine
from lightllm.models.deepseek_v4_dspark.layer_infer.post_layer_infer import DeepseekV4DSparkPostLayerInfer


class DeepseekV41DSparkPostLayerInfer(DeepseekV4DSparkPostLayerInfer):
    def _mix_hc_streams(self, input_embdings, layer_weight):
        streams, pre_mix = input_embdings
        return hc_combine(streams, pre_mix)
