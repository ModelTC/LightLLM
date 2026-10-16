"""PyTorch Metal backend for linear-attention recurrent steps."""

from lightllm.common.basemodel.attention.linear.gdn import LinearAttBackend
from .kernels import (
    fused_recurrent_gated_delta_rule,
)


def _recurrent_prefill(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    output, final_state = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        inplace_final_state=False,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    return (output, final_state) if output_final_state else output


class MetalLinearAttBackend(LinearAttBackend):
    """Use Torch recurrent operations for GDN prefill and decode on Metal."""

    def get_prefill_kernel(self):
        return _recurrent_prefill


def get_qwen35_linear_prefill_att_backend_class(index=1, priority_list=None):
    return MetalLinearAttBackend


def get_qwen35_linear_decode_att_backend_class(index=1, priority_list=None):
    return MetalLinearAttBackend
