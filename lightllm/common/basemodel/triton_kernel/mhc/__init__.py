# SPDX-License-Identifier: Apache-2.0

"""mHC：将残差扩展为 S 条 stream，在每个 attention / FFN 子层前合并、子层后更新。

T = tokens，S = streams，H = 每条 stream 的 hidden size，K = S * H，M = 2 * S + S * S。
x: [T, K] 为当前子层的残差状态，R = x.view(T, S, H)。
fn: [M, K]，scale: [3]，base: [M] 为可学习参数。

1. hc_expand：把 embedding [T, H] 复制 S 份，得到初始 x [T, K]。

2. hc_pre_norm：生成混合权重、合并 stream，并完成子层 RMSNorm：
       inv_rms = rsqrt(mean(x.float() ** 2, dim=-1, keepdim=True) + rms_eps)
       mixes = (x.float() @ fn.T) * inv_rms
   mixes 按 S、S、S*S 拆成 pre_raw、post_raw、residual_raw，最后一组 view 为 [T, S, S]：
       pre = sigmoid(pre_raw * scale[0] + base[:S]) + hc_eps
       post_mix = post_multiplier * sigmoid(post_raw * scale[1] + base[S:2*S])
       residual_logits = residual_raw * scale[2] + base[2*S:].view(S, S)
   residual_logits 经 softmax 和 Sinkhorn 得到 residual_mix [T, S, S]，行列和接近 1。
   用 pre [T, S] 合并残差，得到子层输入：
       layer_input[t, h] = sum_i pre[t, i] * R[t, i, h]  # [T, H]
   合并结果先转 bf16，再在 H 维做带 norm_weight 的 RMSNorm。
   投影和平方和默认由 DeepGEMM 计算；LIGHTLLM_DISABLE_DEEPGEMM_MHC=1 改用 PyTorch。
   两种后端共用后续的 Triton 融合 kernel。

3. hc_post：用子层输出 y [T, H] 更新残差：
       out[t, j, h] = post_mix[t, j] * y[t, h] + sum_i residual_mix[t, i, j] * R[t, i, h]
   out 展平为 [T, K]，作为下一子层的 x；i、j 分别是输入、输出 stream。

4. hc_contract：模型末尾沿 S 维取均值，[T, K] -> [T, H]。

hc_pre_norm / hc_post 仅支持 S = 4；hc_pre_norm 要求 bf16 激活和 fp32 fn。
"""

from .post import hc_post
from .pre_norm import hc_pre_norm
from .streams import hc_contract, hc_expand

__all__ = [
    "hc_contract",
    "hc_expand",
    "hc_post",
    "hc_pre_norm",
]
