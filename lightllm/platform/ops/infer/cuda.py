import torch
from typing import Any, Callable, Optional
from lightllm.common.basemodel.triton_kernel.multimodal_emb import multimodal_emb as cuda_multimodal_emb
from lightllm.server.embed_cache.copy_to_cache import offload_embed_tensor_to_cache
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.platform.base.ops import InferOps


class CudaInferOps(InferOps):

    def multimodal_emb(
        self,
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

    def offload_embed_tensor_to_cache(
        self,
        *,
        embed_tensor: torch.Tensor,
        cache_tensor: torch.Tensor,
        start_index_in_cache: int,
    ) -> None:
        offload_embed_tensor_to_cache(
            embed_tensor=embed_tensor,
            cache_tensor=cache_tensor,
            start_index_in_cache=start_index_in_cache,
        )

    def rotary_emb(
        self,
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

    def ffn(
        self,
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
