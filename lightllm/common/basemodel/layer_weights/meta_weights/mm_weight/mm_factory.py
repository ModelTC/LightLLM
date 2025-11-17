from lightllm.common.quantization import Quantcfg
from lightllm.common.quantization.quantize_method import QuantizationMethod
from typing import Type, Union, Dict
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import (
    MMWeightTpl,
    BMMWeightTpl,
)
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.rowmm_weight import (
    StandardROWMMWeight,
    UnquantizedROWBMMWeight,
    ROWMM_WEIGHT_CLS_MAP,
)
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.colmm_weight import (
    StandardCOLMMWeight,
    COLMM_WEIGHT_CLS_MAP,
)


class MMWeight:
    def __new__(cls, **kwargs):
        """
        weight_names,
        data_type,
        bias_names,
        quant_cfg,
        layer_num,
        name,
        tp_rank,
        tp_world_size,
        ...
        该类主要是通过重载 __new__ 为对应的mm权重绑定量化方法，其他参数都是透传。
        """

        quant_cfg = kwargs.pop("quant_cfg", None)
        layer_num_ = kwargs.pop("layer_num", None)
        name = kwargs.pop("name", None)
        quant_method, quantized_weight = cls._get_quant_method(quant_cfg, layer_num_, name)
        # quantized_weight 本身是用来标识权重本身在文件中是否是以量化后的形式存储，
        # 现在不再使用该参数，是否量化由后续的加载过程自动识别。
        kwargs["quant_method"] = quant_method
        mmcls = cls._get_mmcls(quant_method)
        return mmcls(**kwargs)

    @classmethod
    def _get_quant_method(cls, quant_cfg: Quantcfg, layer_num_: int, name: str) -> QuantizationMethod:
        if quant_cfg is None:
            return None, False
        quant_method: QuantizationMethod = quant_cfg.get_quant_method(layer_num_, name)
        if quant_method is None:
            return None, False
        quant_method.hf_quantization_config = quant_cfg.hf_quantization_config
        quantized_weight = quant_cfg.quantized_weight
        return quant_method, quantized_weight

    @classmethod
    def _get_mmcls(cls, quant_method: QuantizationMethod) -> Type[Union[MMWeightTpl, BMMWeightTpl]]:
        raise NotImplementedError("Subclasses must implement _get_mmcls method")


class ROWMMWeight(MMWeight):
    @classmethod
    def _get_mmcls(cls, quant_method: QuantizationMethod):
        if quant_method is None:
            return StandardROWMMWeight

        return ROWMM_WEIGHT_CLS_MAP.get(
            quant_method.method_name,
            StandardROWMMWeight,
        )


class ROWBMMWeight(MMWeight):
    @classmethod
    def _get_mmcls(cls, quant_method: QuantizationMethod):
        if quant_method is None:
            return UnquantizedROWBMMWeight
        else:
            # TODO: Implement more quantization weight
            raise NotImplementedError("ROWBMMWeight is not implemented")


class COLMMWeight(MMWeight):
    @classmethod
    def _get_mmcls(cls, quant_method: QuantizationMethod):
        if quant_method is None:
            return StandardCOLMMWeight
        return COLMM_WEIGHT_CLS_MAP.get(
            quant_method.method_name,
            StandardCOLMMWeight,
        )
