from lightllm.common.attention.api import VarlenAttentionBackend
from lightllm.common.attention.selection import get_varlen_backend_class, select_varlen_attention_backend
from lightllm.platform import get_device_runtime

VIT_ATTN_BACKEND: VarlenAttentionBackend = None


class _InitializingVarlenAttention:
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
        name = select_varlen_attention_backend(
            self.requested, device=inputs.query.device, dtype=inputs.query.dtype, head_dim=inputs.query.shape[-1]
        )
        backend = get_varlen_backend_class(name)()
        global VIT_ATTN_BACKEND
        VIT_ATTN_BACKEND = backend
        self.forward = backend.forward
        return self.forward(inputs, out=out)


def set_vit_att_backend(backend_name: str, *, defer_validation=False):
    global VIT_ATTN_BACKEND
    VIT_ATTN_BACKEND = (
        _InitializingVarlenAttention(backend_name) if defer_validation else get_varlen_backend_class(backend_name)()
    )
    return


def get_vit_attn_backend():
    if VIT_ATTN_BACKEND is None:
        raise RuntimeError("VIT_ATTN_BACKEND is not initialized. Call set_vit_att_backend() first.")
    return VIT_ATTN_BACKEND
