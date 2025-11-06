from lightllm.common.quantization import Quantcfg
from lightllm.common.quantization.quantize_method import QuantizationMethod
from typing import Type, Union, Dict
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import (
    MMWeightTpl,
    MultiMMWeightTpl,
    BMMWeightTpl,
)
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.rowmm_weight import (
    UnquantizedROWMMWeight,
    UnquantizedROWBMMWeight,
    UnquantizedMultiROWMMWeight,
    ROWMM_WEIGHT_CLS_MAP,
    MULTI_ROWMM_WEIGHT_CLS_MAP,
)
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.colmm_weight import (
    UnquantizedCOLMMWeight,
    COLMM_WEIGHT_CLS_MAP,
)


class MMWeight:
    def __new__(cls, **kwargs):
        quant_cfg = kwargs.pop("quant_cfg", None)
        layer_num_ = kwargs.pop("layer_num", None)
        name = kwargs.pop("name", None)
        quant_method, quantized_weight = cls._get_quant_method(quant_cfg, layer_num_, name)
        kwargs["quant_method"] = quant_method
        mmcls = cls._get_mmcls(quant_method, quantized_weight)
        return mmcls(**kwargs)

    @classmethod
    def _get_quant_method(cls, quant_cfg: Quantcfg, layer_num_: int, name: str) -> QuantizationMethod:
        if quant_cfg is None:
            return None, False
        quant_method = quant_cfg.get_quant_method(layer_num_, name)
        if quant_method is None:
            return None, False
        quant_method.hf_quantization_config = quant_cfg.hf_quantization_config
        quantized_weight = quant_cfg.quantized_weight
        return quant_method, quantized_weight

    @classmethod
    def _get_mmcls(
        cls, quant_method: QuantizationMethod, quantized_weight: bool
    ) -> Type[Union[MMWeightTpl, MultiMMWeightTpl, BMMWeightTpl]]:
        raise NotImplementedError("Subclasses must implement _get_mmcls method")


class ROWMMWeight(MMWeight):
    @classmethod
    def _get_mmcls(cls, quant_method: QuantizationMethod, quantized_weight: bool):
        if quant_method is None or not quantized_weight:
            return UnquantizedROWMMWeight

        return ROWMM_WEIGHT_CLS_MAP[quant_method.method_name]


class MultiROWMMWeight(MMWeight):
    @classmethod
    def _get_mmcls(cls, quant_method: QuantizationMethod, quantized_weight: bool):
        if quant_method is None or not quantized_weight:
            return UnquantizedMultiROWMMWeight

        return MULTI_ROWMM_WEIGHT_CLS_MAP[quant_method.method_name]


class ROWBMMWeight(MMWeight):
    @classmethod
    def _get_mmcls(cls, quant_method: QuantizationMethod, quantized_weight: bool):
        if quant_method is None or not quantized_weight:
            return UnquantizedROWBMMWeight
        else:
            # TODO: Implement more quantization weight
            raise NotImplementedError("ROWBMMWeight is not implemented")


class COLMMWeight(MMWeight):
    @classmethod
    def _get_mmcls(cls, quant_method: QuantizationMethod, quantized_weight: bool):
        if quant_method is None or not quantized_weight:
            return UnquantizedCOLMMWeight
        return COLMM_WEIGHT_CLS_MAP[quant_method.method_name]
