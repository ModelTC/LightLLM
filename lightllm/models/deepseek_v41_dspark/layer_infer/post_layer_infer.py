from lightllm.models.deepseek_v4_dspark.layer_infer.post_layer_infer import DeepseekV4DSparkPostLayerInfer


class DeepseekV41DSparkPostLayerInfer(DeepseekV4DSparkPostLayerInfer):
    def _mix_hc_streams(self, input_embdings, layer_weight):
        streams, pre_mix = input_embdings
        streams = streams.view(streams.shape[0], pre_mix.shape[1], -1)
        return (streams.float() * pre_mix[..., None]).sum(1).to(streams.dtype)
