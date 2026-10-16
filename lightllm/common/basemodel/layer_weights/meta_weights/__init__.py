from importlib import import_module

_EXPORTS = {
    "BaseWeight": (".base_weight", "BaseWeight"),
    "MMWeightTpl": (".mm_weight", "MMWeightTpl"),
    "ROWMMWeight": (".mm_weight", "ROWMMWeight"),
    "KVROWNMMWeight": (".mm_weight", "KVROWNMMWeight"),
    "ROWBMMWeight": (".mm_weight", "ROWBMMWeight"),
    "QKVROWNMMWeight": (".mm_weight", "QKVROWNMMWeight"),
    "COLMMWeight": (".mm_weight", "COLMMWeight"),
    "TpRMSNormWeight": (".norm_weight", "TpRMSNormWeight"),
    "RMSNormWeight": (".norm_weight", "RMSNormWeight"),
    "GatedRMSNormWeight": (".norm_weight", "GatedRMSNormWeight"),
    "LayerNormWeight": (".norm_weight", "LayerNormWeight"),
    "NoTpGEMMANormWeight": (".norm_weight", "NoTpGEMMANormWeight"),
    "QKRMSNORMWeight": (".norm_weight", "QKRMSNORMWeight"),
    "QKGEMMANormWeight": (".norm_weight", "QKGEMMANormWeight"),
    "EmbeddingWeight": (".embedding_weight", "EmbeddingWeight"),
    "LMHeadWeight": (".embedding_weight", "LMHeadWeight"),
    "NoTpPosEmbeddingWeight": (".embedding_weight", "NoTpPosEmbeddingWeight"),
    "TpAttSinkWeight": (".att_sink_weight", "TpAttSinkWeight"),
    "FusedMoeWeight": (".fused_moe.fused_moe_weight", "FusedMoeWeight"),
    "ParameterWeight": (".parameter_weight", "ParameterWeight"),
    "TpParameterWeight": (".parameter_weight", "TpParameterWeight"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attr = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attr)
    globals()[name] = value
    return value
