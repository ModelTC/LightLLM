import torch
from typing import Any, Callable, Optional, Tuple

from lightllm.common.basemodel.triton_kernel.embedding import npu_embedding
from lightllm.common.basemodel.triton_kernel.multimodal_emb import npu_multimodal_emb
from lightllm.models.llama.layer_infer.transformer_layer_infer import npu_ffn_fwd
from lightllm.models.llama.triton_kernel.rotary_emb import npu_rotary_emb_fwd
from lightllm.platform.base.ops import register_op, out_like
from lightllm.server.embed_cache.copy_to_cache import npu_offload_embed_tensor_to_cache


@register_op("ascend")
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
    npu_multimodal_emb(
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


@register_op("ascend")
def offload_embed_tensor_to_cache(
    *,
    embed_tensor: torch.Tensor,
    cache_tensor: torch.Tensor,
    start_index_in_cache: int,
) -> None:
    npu_offload_embed_tensor_to_cache(
        embed_tensor=embed_tensor,
        cache_tensor=cache_tensor,
        start_index_in_cache=start_index_in_cache,
    )


@register_op("ascend")
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
    impl = rotary_impl or npu_rotary_emb_fwd
    impl(
        is_prefill=is_prefill,
        batch_size=batch_size,
        q=q,
        k=k,
        cos=cos,
        sin=sin,
        partial_rotary_factor=partial_rotary_factor,
    )


@register_op("ascend")
def ffn(
    *,
    input: torch.Tensor,
    layer_weight: Any,
    alloc_func: Callable,
    embed_dim: int,
) -> torch.Tensor:
    return npu_ffn_fwd(input, layer_weight, embed_dim)


@register_op(
    "ascend", 
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
    npu_embedding(input_ids, weight, vob_start_id, vob_end_id, out)
    return out


@register_op(
    "ascend", 
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


@register_op("ascend", out=out_like("input"))
def rms_norm(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    gate_value: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if gate_value is not None:
        raise NotImplementedError("gate_value is not supported for rms_norm on ascend")

    import torch_npu

    _out = torch_npu.npu_rms_norm(input, weight, epsilon=eps)[0]
    out.copy_(_out)
    return out


@register_op("ascend", out=out_like("input"))
def layer_norm(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    out: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError("layer_norm is not supported on ascend")


@register_op("ascend")
def qk_rms_norm(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    eps: float,
    fp32_multiply: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    import torch_npu

    head_dim_q = w_q.shape[0]
    head_dim_k = w_k.shape[0]
    flat_q = q.reshape(-1, head_dim_q)
    flat_k = k.reshape(-1, head_dim_k)
    _q = torch_npu.npu_rms_norm(flat_q, w_q, epsilon=eps)[0]
    _k = torch_npu.npu_rms_norm(flat_k, w_k, epsilon=eps)[0]
    _q = _q.view(q.shape)
    _k = _k.view(k.shape)
    q.copy_(_q)
    k.copy_(_k)

    return q, k
