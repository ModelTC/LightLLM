"""Run one cached attention query through the serving adapter during provider selection."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from lightllm.common.basemodel.basemodel import TpPartBaseModel


@dataclass(frozen=True)
class AttentionProbe:
    family: str
    phase: str
    dtype: torch.dtype
    q_heads: int
    kv_heads: int
    head_dim: int
    page_size: int
    nope_dim: int = 0
    rope_dim: int = 0
    latent_dim: int = 0
    value_dim: int = 0
    topk: int = 0

    @classmethod
    def from_model(cls, model: "TpPartBaseModel", family: str, phase: str) -> "AttentionProbe":
        dims = {}
        if family in ("mla", "sparse_mla"):
            dims = dict(
                nope_dim=model.qk_nope_head_dim,
                rope_dim=model.qk_rope_head_dim,
                latent_dim=model.kv_lora_rank,
                value_dim=model.v_head_dim,
                topk=model.config.get("index_topk", 0),
            )
        return cls(
            family=family,
            phase=phase,
            dtype=model.data_type,
            q_heads=model.config["num_attention_heads"] // model.tp_world_size_,
            kv_heads=max(1, model.tp_k_head_num_),
            head_dim=model.head_dim_,
            page_size=model.args.page_size,
            **dims,
        )

    def run(self, backend_path: str, device: torch.device) -> None:
        import torch
        from lightllm.common.basemodel.attention.base_att import AttControl
        from lightllm.common.basemodel.attention.import_utils import load_backend_class
        from lightllm.common.basemodel.infer_struct import InferStateInfo
        from lightllm.common.basemodel.layer_infer.base_layer_infer import BaseLayerInfer

        # A single query after one cached token exercises the paged/cache path in both phases.
        state = InferStateInfo()
        offsets = torch.arange(3, device=device, dtype=torch.int32)
        state.batch_size, state.total_token_num = 1, 2
        state.max_q_seq_len, state.max_kv_seq_len, state.max_cache_len = 1, 2, 1
        state.b_req_idx = state.b_q_start_loc = state.b_kv_start_loc = offsets[:1]
        state.b_seq_len = offsets[2:]
        state.b_q_seq_len = state.b_ready_cache_len = state.mem_index = offsets[1:2]
        state.b1_cu_q_seq_len, state.b1_cu_kv_seq_len = offsets[:2], offsets[::2].contiguous()
        state.input_ids = offsets[:1].long()
        state.is_prefill = self.phase == "prefill"
        slots = max(2, self.page_size)
        state.req_manager = SimpleNamespace(
            req_to_token_indexs=torch.arange(slots, dtype=torch.int32, device=device)[None], HOLD_REQUEST_ID=1
        )
        backend = load_backend_class(backend_path)(model=_ProbeModel(self, state.req_manager, slots))
        prefill = self.phase == "prefill"
        att = (
            backend.create_att_prefill_state(infer_state=state)
            if prefill
            else backend.create_att_decode_state(infer_state=state)
        )
        att.init_state()

        opts = dict(device=device, dtype=self.dtype)
        scale = (self.head_dim if self.family == "gqa" else self.nope_dim + self.rope_dim) ** -0.5
        control = AttControl()
        if self.family == "gqa":
            q = torch.zeros((1, self.q_heads, self.head_dim), **opts)
            k = torch.zeros((slots, self.kv_heads, self.head_dim), **opts)
            k[1, :, -1] = 1
            v = torch.randn_like(k)
            values = v[:2].repeat_interleave(self.q_heads // self.kv_heads, dim=1)
        elif self.family == "mla" and prefill:
            q = torch.zeros((1, self.q_heads, self.nope_dim + self.rope_dim), **opts)
            k = (torch.zeros((2, self.q_heads, self.nope_dim), **opts), torch.zeros((2, 1, self.rope_dim), **opts))
            k[1][1, :, -1] = 1
            v = values = torch.randn((2, self.q_heads, self.value_dim), **opts)
            control = AttControl(mla_prefill=True, mla_prefill_dict={"softmax_scale": scale})
        elif self.family == "mla":
            q = (
                torch.zeros((1, self.q_heads, self.latent_dim), **opts),
                torch.zeros((1, self.q_heads, self.rope_dim), **opts),
            )
            k = torch.randn((slots, 1, self.latent_dim + self.rope_dim), **opts)
            k[0, :, -1], k[1, :, -1] = 0, 1
            v = None
            values = k[:2, :, : self.latent_dim].expand(-1, self.q_heads, -1)
            control = AttControl(mla_decode=True, mla_decode_dict={"softmax_scale": scale})
        else:
            # DeepSeek 3.2 stores one FP32 scale for each group of 128 FP8 latent values.
            q = torch.zeros((1, self.q_heads, self.latent_dim + self.rope_dim), **opts)
            latent = torch.randn((2, 1, self.latent_dim), **opts).to(torch.float8_e4m3fn)
            rope = torch.zeros((2, 1, self.rope_dim), **opts)
            rope[1, :, -1] = 1
            scales = torch.ones((2, 1, self.latent_dim // 128), dtype=torch.float32, device=device)
            k = torch.cat((latent.view(torch.uint8), scales.view(torch.uint8), rope.view(torch.uint8)), dim=-1)
            v = None
            values = latent.to(self.dtype).expand(-1, self.q_heads, -1)
            indices = torch.full((1, self.topk), -1, dtype=torch.int32, device=device)
            indices[0, :2] = offsets[:2]
            params = dict(topk_mem_indices=indices, softmax_scale=scale, kv_lora_rank=self.latent_dim)
            if prefill:
                current_kv = torch.cat((latent[1:].to(self.dtype), rope[1:]), dim=-1)
                control = AttControl(
                    nsa_prefill=True, nsa_prefill_dict=dict(params, topk_indices=indices, prefill_cache_kv=current_kv)
                )
            else:
                q = q.split((self.latent_dim, self.rope_dim), dim=-1)
                control = AttControl(nsa_decode=True, nsa_decode_dict=params)

        rotary_q = q[1] if isinstance(q, tuple) else q
        rotary_q[..., -1] = 8
        run = att.prefill_att if prefill else att.decode_att
        out = run(q=q, k=k, v=v, att_control=control, alloc_func=BaseLayerInfer().alloc_tensor)
        # Only the second key has a nonzero dot product with the query.
        weight = 1.0 / (1.0 + exp(-8 * scale))
        expected = torch.lerp(values[0].float(), values[1].float(), weight).unsqueeze(0).to(self.dtype)
        torch.testing.assert_close(out, expected, atol=0.01, rtol=0.01)


class _ProbeModel:
    """Geometry needed to construct a serving adapter in the isolated validation process."""

    graph = None
    graph_max_batch_size = 1
    is_mtp_draft_model = False

    def __init__(self, case: AttentionProbe, req_manager: SimpleNamespace, slots: int) -> None:
        from copy import copy
        from lightllm.common.basemodel.mtp_manager import MtpManager
        from lightllm.utils.dist_utils import get_dp_world_size
        from lightllm.utils.envs_utils import get_env_start_args

        self.args = copy(get_env_start_args())
        self.args.running_max_req_size = 1
        self.tp_world_size_ = get_dp_world_size()
        heads = case.q_heads * self.tp_world_size_
        self.config = dict(
            num_attention_heads=heads,
            num_key_value_heads=case.kv_heads * self.tp_world_size_,
            head_dim=case.head_dim,
            hidden_size=heads * case.head_dim,
            rope_scaling=None,
        )
        self.data_type = case.dtype
        self.req_manager = req_manager
        self.max_seq_length = self.graph_max_len_in_batch = slots
        self.qk_nope_head_dim, self.qk_rope_head_dim = case.nope_dim, case.rope_dim
        self.kv_lora_rank, self.v_head_dim = case.latent_dim, case.value_dim
        # Hardware validation exercises ordinary attention regardless of speculative batch layout.
        self.mtp_manager = MtpManager()
        self.mtp_manager.args = copy(self.args)
        self.mtp_manager.args.mtp_mode = None
