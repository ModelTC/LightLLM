"""Metal projection specialization retaining LightLLM's weight interfaces."""
from lightllm.common.quantization.no_quant import NoQuantization
from lightllm.common.quantization.registry import QUANTMETHODS
from .gemv import project as project_vector
from .gemm import project as project_matrix
from .batched_gemv import project as project_batch
from .simd_projection import project as project_simd


@QUANTMETHODS.register("none", platform="metal")
class MetalNoQuantization(NoQuantization):
    def apply(self, input_tensor, weight_pack, out=None, workspace=None, use_custom_tensor_mananger=True, bias=None):
        result = project_vector(input_tensor, weight_pack.weight, out=out, bias=bias)
        if result is None:
            result = project_simd(input_tensor, weight_pack.weight, out=out, bias=bias)
        if result is None:
            result = project_batch(input_tensor, weight_pack.weight, out=out, bias=bias)
        if result is None:
            result = project_matrix(input_tensor, weight_pack.weight, out=out, bias=bias)
        if result is not None:
            return result
        if (
            out is not None
            and input_tensor.device.type == "mps"
            and input_tensor.ndim == weight_pack.weight.ndim == 2
            and out.shape == (input_tensor.shape[0], weight_pack.weight.shape[0])
            and out.dtype == input_tensor.dtype
            and out.device == input_tensor.device
            and (out.storage_offset() != 0 or not out.is_contiguous())
        ):
            # Torch MPS mm(out=...) can leave an offset output view untouched.
            # Compute into an ordinary allocation, then honor the caller's view.
            result = super().apply(input_tensor, weight_pack, None, workspace, use_custom_tensor_mananger, bias)
            out.copy_(result)
            return out
        return super().apply(input_tensor, weight_pack, out, workspace, use_custom_tensor_mananger, bias)
