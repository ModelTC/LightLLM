import torch

from lightllm.common.quantization.no_quant import NoQuantization
from lightllm.common.quantization.quantize_method import WeightPack


class OutputAlignedProjection(NoQuantization):
    def __init__(self, alignment=8):
        super().__init__()
        if type(alignment) is not int or alignment < 1 or alignment & (alignment - 1):
            raise ValueError("Projection alignment must be a positive power of two")
        self.alignment = alignment
        self._padded_weight = None

    def _create_weight(self, out_dims, in_dim, dtype, device_id, num_experts=1):
        width = sum(out_dims) if isinstance(out_dims, list) else out_dims
        physical_width = (width + self.alignment - 1) // self.alignment * self.alignment
        self._padded_weight = None
        if dtype not in (torch.float16, torch.bfloat16) or physical_width == width or num_experts != 1:
            return super()._create_weight(out_dims, in_dim, dtype, device_id, num_experts)
        self._padded_weight = torch.zeros((physical_width, in_dim), dtype=dtype, device=device_id)
        pack = WeightPack(weight=self._padded_weight[:width])
        parts = self._split_weight_pack(pack, out_dims if isinstance(out_dims, list) else [out_dims], -2)
        return pack, parts

    def apply(self, input_tensor, weight_pack, out=None, workspace=None, use_custom_tensor_mananger=True, bias=None):
        if self._padded_weight is None or bias is not None or input_tensor.shape[0] == 0:
            return super().apply(
                input_tensor,
                weight_pack,
                out,
                workspace,
                use_custom_tensor_mananger=use_custom_tensor_mananger,
                bias=bias,
            )
        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        def allocate(shape):
            if use_custom_tensor_mananger:
                return g_cache_manager.alloc_tensor(shape, input_tensor.dtype, device=input_tensor.device)
            return torch.empty(shape, dtype=input_tensor.dtype, device=input_tensor.device)

        rows, width = input_tensor.shape[0], weight_pack.weight.shape[0]
        physical_output = allocate((rows, self._padded_weight.shape[0]))
        torch.mm(input_tensor, self._padded_weight.T, out=physical_output)
        if out is None:
            out = allocate((rows, width))
        elif out.shape != (rows, width) or out.dtype != input_tensor.dtype or out.device != input_tensor.device:
            raise ValueError("Projection output must match logical shape, dtype and device")
        out.copy_(physical_output[:, :width])
        return out
