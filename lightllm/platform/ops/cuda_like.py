import torch
from typing import Any, Callable, Optional, Tuple

from lightllm.common.basemodel.triton_kernel.embedding import embedding as cuda_embedding
from lightllm.common.basemodel.triton_kernel.multimodal_emb import multimodal_emb as cuda_multimodal_emb
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.common.basemodel.triton_kernel.norm.gated_rmsnorm import gated_rmsnorm_forward
from lightllm.common.basemodel.triton_kernel.norm.layernorm import layernorm_forward
from lightllm.common.basemodel.triton_kernel.norm.qk_norm import qk_rmsnorm_fused_forward
from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.platform.base.ops import register_op, out_like
from lightllm.server.embed_cache.copy_to_cache import (
    offload_embed_tensor_to_cache as cuda_offload_embed_tensor_to_cache,
)


@register_op("cuda_like")
def multimodal_emb(
    *,
    out: torch.Tensor,
    prompt_ids: torch.Tensor,
    text_weight_embs: torch.Tensor,
    embed_cache: torch.Tensor,
    img_token_lens: torch.Tensor,
    img_start_token_ids: torch.Tensor,
    img_start_locs_in_cache: torch.Tensor,
    tp_text_start_token_id: int,
    tp_text_end_token_id: int,
    tp_world_size: int,
) -> None:
    cuda_multimodal_emb(
        out=out,
        prompt_ids=prompt_ids,
        text_weight_embs=text_weight_embs,
        embed_cache=embed_cache,
        img_token_lens=img_token_lens,
        img_start_token_ids=img_start_token_ids,
        img_start_locs_in_cache=img_start_locs_in_cache,
        tp_text_start_token_id=tp_text_start_token_id,
        tp_text_end_token_id=tp_text_end_token_id,
        tp_world_size=tp_world_size,
    )


@register_op("cuda_like")
def offload_embed_tensor_to_cache(
    *,
    embed_tensor: torch.Tensor,
    cache_tensor: torch.Tensor,
    start_index_in_cache: int,
) -> None:
    cuda_offload_embed_tensor_to_cache(
        embed_tensor=embed_tensor,
        cache_tensor=cache_tensor,
        start_index_in_cache=start_index_in_cache,
    )


@register_op("cuda_like")
def rotary_emb(
    *,
    is_prefill: bool,
    batch_size: int,
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    partial_rotary_factor: float = 1.0,
    rotary_impl: Optional[Callable] = None,
) -> None:
    impl = rotary_impl or rotary_emb_fwd
    impl(q=q, k=k, cos=cos, sin=sin, partial_rotary_factor=partial_rotary_factor)


@register_op("cuda_like")
def ffn(
    *,
    input: torch.Tensor,
    layer_weight: Any,
    alloc_func: Callable,
    embed_dim: int,
) -> torch.Tensor:
    input = input.view(-1, embed_dim)
    up_gate_out = layer_weight.gate_up_proj.mm(input)
    ffn1_out = alloc_func(
        (input.size(0), up_gate_out.size(1) // 2),
        dtype=input.dtype,
        device=input.device,
    )
    silu_and_mul_fwd(up_gate_out, ffn1_out)
    return layer_weight.down_proj.mm(ffn1_out)


@register_op(
    "cuda_like", 
    out=lambda kwargs: (
        (kwargs["input_ids"].shape[0], kwargs["weight"].shape[1]),
        kwargs["weight"].dtype,
        kwargs["weight"].device,
    ),
)
def embedding(
    *,
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    vob_start_id: int,
    vob_end_id: Optional[int] = None,
) -> torch.Tensor:
    if vob_end_id is None:
        vob_end_id = weight.shape[0]
    cuda_embedding(
        input_ids=input_ids,
        weight=weight,
        vob_start_id=vob_start_id,
        vob_end_id=vob_end_id,
        out=out,
    )
    return out


@register_op(
    "cuda_like", 
    out=lambda kwargs: (
        (kwargs["weight"].shape[0], kwargs["input"].shape[1]),
        kwargs["weight"].dtype,
        kwargs["weight"].device,
    ),
)
def lm_head(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    torch.mm(weight, input, out=out)
    return out


@register_op("cuda_like", out=out_like("input"))
def rms_norm(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    gate_value: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if gate_value is None:
        rmsnorm_forward(x=input, weight=weight, eps=eps, out=out)
    else:
        gated_rmsnorm_forward(x=input, weight=weight, bias=None, eps=eps, z=gate_value, out=out)
    return out


@register_op("cuda_like", out=out_like("input"))
def layer_norm(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    out: torch.Tensor,
) -> torch.Tensor:
    out_ = layernorm_forward(x=input, weight=weight, bias=bias, eps=eps)
    out.copy_(out_)
    return out


@register_op("cuda_like")
def qk_rms_norm(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    eps: float,
    fp32_multiply: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return qk_rmsnorm_fused_forward(q=q, k=k, w_q=w_q, w_k=w_k, eps=eps, fp32_multiply=fp32_multiply)
