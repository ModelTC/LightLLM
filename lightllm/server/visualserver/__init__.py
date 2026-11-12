from lightllm.common.attention.api import VarlenAttentionOp
from lightllm.common.attention.selection import get_varlen_attention_op_class, select_varlen_attention_op
from lightllm.platform import get_device_runtime

VIT_ATTENTION_OP: VarlenAttentionOp = None


class _InitializingVarlenAttentionOp:
    """用 worker 预热时的真实 Q/K/V 选择实现，之后直接绑定实现的 forward。

    部分视觉模型固定使用 BF16，与 llm 的 data_type 不同。manager 无法据启动
    参数推断实际 dtype/head dim，因此选择发生在各设备的首次预热输入到达时。
    """

    def __init__(self, requested):
        self.requested = requested

    def forward(self, inputs, *, out):
        runtime = get_device_runtime(inputs.query.device)
        if runtime.is_capturing():
            raise RuntimeError("warm up vision attention before Graph capture")
        name = select_varlen_attention_op(
            self.requested, device=inputs.query.device, dtype=inputs.query.dtype, head_dim=inputs.query.shape[-1]
        )
        op = get_varlen_attention_op_class(name)()
        global VIT_ATTENTION_OP
        VIT_ATTENTION_OP = op
        self.forward = op.forward
        return self.forward(inputs, out=out)


def set_vit_attention_op(provider: str, *, defer_validation=False):
    global VIT_ATTENTION_OP
    VIT_ATTENTION_OP = (
        _InitializingVarlenAttentionOp(provider) if defer_validation else get_varlen_attention_op_class(provider)()
    )
    return


def get_vit_attention_op():
    if VIT_ATTENTION_OP is None:
        raise RuntimeError("VIT_ATTENTION_OP is not initialized. Call set_vit_attention_op() first.")
    return VIT_ATTENTION_OP
