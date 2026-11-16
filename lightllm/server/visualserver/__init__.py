from lightllm.common.attention.api import VarlenAttentionOp
from lightllm.common.attention.registry import get_varlen_attention_op_class
from lightllm.common.attention.selection import select_varlen_attention_op
from lightllm.platform import get_device_runtime


class _LazyVarlenAttentionRunner:
    """用 worker 预热时的真实 Q/K/V 选择实现，之后通过固定的 forward 委托执行。

    部分视觉模型固定使用 BF16，与 llm 的 data_type 不同。manager 无法据启动
    参数推断实际 dtype/head dim，因此选择发生在各设备的首次预热输入到达时。
    """

    def __init__(self, requested):
        self.requested = requested
        self._op = None

    def forward(self, inputs, *, out):
        if self._op is None:
            runtime = get_device_runtime(inputs.query.device)
            if runtime.is_capturing():
                raise RuntimeError("warm up vision attention before Graph capture")
            name = select_varlen_attention_op(
                self.requested, device=inputs.query.device, dtype=inputs.query.dtype, head_dim=inputs.query.shape[-1]
            )
            self._op = get_varlen_attention_op_class(name)()
        return self._op.forward(inputs, out=out)


VIT_ATTENTION_OP: VarlenAttentionOp | _LazyVarlenAttentionRunner | None = None


def set_vit_attention_op(provider: str, *, defer_validation=False):
    global VIT_ATTENTION_OP
    VIT_ATTENTION_OP = (
        _LazyVarlenAttentionRunner(provider) if defer_validation else get_varlen_attention_op_class(provider)()
    )
    return


def get_vit_attention_op():
    if VIT_ATTENTION_OP is None:
        raise RuntimeError("VIT_ATTENTION_OP is not initialized. Call set_vit_attention_op() first.")
    return VIT_ATTENTION_OP
