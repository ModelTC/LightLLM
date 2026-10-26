from lightllm.common.basemodel.attention.linear.gdn import GDNLinearAttBackend


class FlashQlaLinearAttBackend(GDNLinearAttBackend):
    def get_prefill_kernel(self):
        from flash_qla import chunk_gated_delta_rule

        return chunk_gated_delta_rule
