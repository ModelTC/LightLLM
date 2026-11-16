from lightllm.common.basemodel.attention.linear.gdn import LinearAttBackend


class FlashQlaLinearAttBackend(LinearAttBackend):
    def get_prefill_kernel(self):
        from flash_qla import chunk_gated_delta_rule

        return chunk_gated_delta_rule
