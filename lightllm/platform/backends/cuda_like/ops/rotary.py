from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.platform.ops.rotary import RotaryOps
from lightllm.platform.plugin.ops import register_ops

register_ops(
    RotaryOps,
    platforms=("cuda", "musa"),
    rotary_emb=rotary_emb_fwd,
)
