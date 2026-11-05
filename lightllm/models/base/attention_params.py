def dense_params(spec, weight, device):
    from lightllm.common.execution.compute.attention.dense.backend import DenseStaticParams

    return DenseStaticParams(spec, device.allocate_tensor, weight.attn_sinks.weight if spec.attention_sink else None)


def mla_params(spec, weight, device):
    from lightllm.common.execution.compute.attention.mla.backend import MLAStaticParams

    return MLAStaticParams(spec, device.allocate_tensor, weight.cc_kv_b_proj_.mm)


def sparse_mla_params(spec, weight, device):
    from lightllm.common.execution.compute.attention.sparse_mla.backend import SparseMLAStaticParams

    return SparseMLAStaticParams(spec, device.allocate_tensor)


def pooled_nope_params(spec, weight, device):
    from lightllm.common.execution.compute.attention.pooled_nope.backend import PooledNoPEStaticParams

    return PooledNoPEStaticParams(spec, device.allocate_tensor, weight.index_kpool_compress_ape.weight)


def gdn_params(spec, weight, device):
    from lightllm.common.execution.compute.attention.gdn.backend import GDNStaticParams

    return GDNStaticParams(
        spec,
        weight.linear_conv1d.mm_param.weight,
        weight.linear_conv1d.bias,
        weight.linear_A_log.weight,
        weight.linear_dt_bias.weight,
    )


def kda_params(spec, weight, device):
    from lightllm.common.execution.compute.attention.kda.backend import KDAStaticParams

    return KDAStaticParams(
        spec, weight.get_merged_kda_conv_weight(), weight.linear_A_log.weight, weight.linear_dt_bias.weight
    )
