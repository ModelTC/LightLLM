import torch

from lightllm.common.basemodel.triton_kernel.scalar_linear import scalar_linear
from lightllm.common.quantization.no_quant import NoQuantization


class StableScalarProjection(NoQuantization):
    def apply(self, input_tensor, weight_pack, out=None, workspace=None, use_custom_tensor_mananger=True, bias=None):
        if not input_tensor.is_cuda or weight_pack.weight.shape[0] != 1 or bias is not None:
            return super().apply(
                input_tensor,
                weight_pack,
                out=out,
                workspace=workspace,
                use_custom_tensor_mananger=use_custom_tensor_mananger,
                bias=bias,
            )
        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        allocate = g_cache_manager.empty if use_custom_tensor_mananger else torch.empty
        return scalar_linear(input_tensor, weight_pack.weight, out=out, alloc_func=allocate)
