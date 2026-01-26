import torch
import torch.nn.functional as F
from lightllm.common.basemodel.attention_vit.base_att import BaseVitAttBackend


class SdpaVitAttBackend(BaseVitAttBackend):
    def vit_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        assert q.ndim == k.ndim == v.ndim == o.ndim == 3
        assert cu_seqlens is not None and cu_seqlens.ndim == 1

        cu = cu_seqlens.to(device=q.device)
        B = cu.numel() - 1

        with torch.no_grad():
            for b in range(B):
                s = int(cu[b].item())
                e = int(cu[b + 1].item())
                L = e - s
                if L <= 0:
                    continue
                if max_seqlen:
                    assert L <= max_seqlen

                # [L, H, D] -> [1, H, L, D]
                q_ = q[s:e].permute(1, 0, 2).unsqueeze(0)
                k_ = k[s:e].permute(1, 0, 2).unsqueeze(0)
                v_ = v[s:e].permute(1, 0, 2).unsqueeze(0)

                out = F.scaled_dot_product_attention(
                    q_,
                    k_,
                    v_,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )
                # [1, H, L, D] -> [L, H, D]
                o[s:e].copy_(out.squeeze(0).permute(1, 0, 2))

        return o
