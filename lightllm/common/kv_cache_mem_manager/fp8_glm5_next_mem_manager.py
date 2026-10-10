from .glm5_next_mem_manager import Glm5NextMemManager
from .operator import LinearAttMemOperator


class FP8Glm5NextMemOperator(LinearAttMemOperator):
    def copy_kv_to_mem_manager(self, layer_index, mem_index, kv):
        from lightllm.models.glm5_next.triton_kernel.destindex_copy_kv_flashmla_fp8 import (
            destindex_copy_kv_flashmla_fp8,
        )

        output = self.mem_manager.get_att_input_params(layer_index)
        destindex_copy_kv_flashmla_fp8(kv, mem_index, output)


class FP8Glm5NextMemManager(Glm5NextMemManager):
    operator_class = FP8Glm5NextMemOperator

    def get_att_input_params(self, layer_index):
        return self._layer_buffer(layer_index)[:, :, : self.linear_config.FP8_MLA_BYTES]
