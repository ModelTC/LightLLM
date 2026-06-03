import torch
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    ROWMMWeight,
    COLMMWeight,
    ROWBMMWeight,
    RMSNormWeight,
    ParameterWeight,
    TpAttSinkWeight,
)
from lightllm.common.basemodel.layer_weights.meta_weights.base_weight import BaseWeightTpl
from lightllm.common.quantization.registry import QUANTMETHODS
from ..triton_kernel.quant_convert import dequant_fp8_block_to_bf16


class DeepseekV4FP4ExpertsWeight(BaseWeightTpl):
    def __init__(self, weight_prefix, n_routed_experts, hidden_size, moe_intermediate_size, data_type):
        super().__init__(data_type=data_type)
        self.weight_prefix = weight_prefix
        self.n_routed_experts = n_routed_experts
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.split_inter_size = moe_intermediate_size // self.tp_world_size_
        self.local_expert_ids = list(range(n_routed_experts))
        self.expert_idx_to_local_idx = {expert_idx: expert_idx for expert_idx in self.local_expert_ids}
        self._create_weight()

    def _create_weight(self):
        device = f"cuda:{self.device_id_}"
        n = self.n_routed_experts
        h = self.hidden_size
        inter = self.split_inter_size
        self.w1 = torch.empty((n, inter, h // 2), dtype=torch.int8, device=device)
        self.w3 = torch.empty((n, inter, h // 2), dtype=torch.int8, device=device)
        self.w2 = torch.empty((n, h, inter // 2), dtype=torch.int8, device=device)
        self.w1_scale = torch.empty((n, inter, h // 32), dtype=torch.float8_e8m0fnu, device=device)
        self.w3_scale = torch.empty((n, inter, h // 32), dtype=torch.float8_e8m0fnu, device=device)
        self.w2_scale = torch.empty((n, h, inter // 32), dtype=torch.float8_e8m0fnu, device=device)
        self.load_ok = {
            name: [False] * n
            for name in ("w1", "w1_scale", "w2", "w2_scale", "w3", "w3_scale")
        }

    def _copy_expert_weight(self, dst, weight, expert_idx, name, is_down=False):
        if is_down:
            start = self.tp_rank_ * self.split_inter_size // 2
            end = (self.tp_rank_ + 1) * self.split_inter_size // 2
            src = weight[:, start:end]
        else:
            start = self.tp_rank_ * self.split_inter_size
            end = (self.tp_rank_ + 1) * self.split_inter_size
            src = weight[start:end, :]
        dst[expert_idx].copy_(src)
        self.load_ok[name][expert_idx] = True

    def _copy_expert_scale(self, dst, scale, expert_idx, name, is_down=False):
        if is_down:
            start = self.tp_rank_ * self.split_inter_size // 32
            end = (self.tp_rank_ + 1) * self.split_inter_size // 32
            src = scale[:, start:end]
        else:
            start = self.tp_rank_ * self.split_inter_size
            end = (self.tp_rank_ + 1) * self.split_inter_size
            src = scale[start:end, :]
        dst[expert_idx].copy_(src)
        self.load_ok[name][expert_idx] = True

    def load_hf_weights(self, weights):
        for expert_idx in self.local_expert_ids:
            prefix = f"{self.weight_prefix}.{expert_idx}"
            w1 = f"{prefix}.w1.weight"
            w1_scale = f"{prefix}.w1.scale"
            w2 = f"{prefix}.w2.weight"
            w2_scale = f"{prefix}.w2.scale"
            w3 = f"{prefix}.w3.weight"
            w3_scale = f"{prefix}.w3.scale"
            if w1 in weights:
                self._copy_expert_weight(self.w1, weights[w1], expert_idx, "w1")
            if w1_scale in weights:
                self._copy_expert_scale(self.w1_scale, weights[w1_scale], expert_idx, "w1_scale")
            if w3 in weights:
                self._copy_expert_weight(self.w3, weights[w3], expert_idx, "w3")
            if w3_scale in weights:
                self._copy_expert_scale(self.w3_scale, weights[w3_scale], expert_idx, "w3_scale")
            if w2 in weights:
                self._copy_expert_weight(self.w2, weights[w2], expert_idx, "w2", is_down=True)
            if w2_scale in weights:
                self._copy_expert_scale(self.w2_scale, weights[w2_scale], expert_idx, "w2_scale", is_down=True)

    def verify_load(self):
        return all(all(ok_list) for ok_list in self.load_ok.values())


class DeepseekV4TransformerLayerWeight(TransformerLayerWeight):
    """Per-layer weights for DeepSeek-V4-Flash.

    The checkpoint stores most linears in FP8 (e4m3 + block-128 ue8m0 scale) and the routed
    experts in FP4 (int8-packed e2m1 + group-32 ue8m0 scale). Hopper does not use the SM100
    MegaMoE path here, so routed experts are kept in packed FP4 and temporarily de-quantized only
    for selected experts in the correctness-first torch MoE path.
    """

    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    def _parse_config(self):
        cfg = self.network_config_
        self.fp8_quant = QUANTMETHODS.get("deepgemm-fp8w8a8-b128")
        self.hidden = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.head_dim = cfg["head_dim"]
        self.rope_dim = cfg["qk_rope_head_dim"]
        self.q_lora_rank = cfg["q_lora_rank"]
        self.o_lora_rank = cfg["o_lora_rank"]
        self.o_groups = cfg["o_groups"]
        self.index_n_heads = cfg["index_n_heads"]
        self.index_head_dim = cfg["index_head_dim"]
        self.n_routed_experts = cfg["n_routed_experts"]
        self.moe_inter = cfg["moe_intermediate_size"]
        self.num_hash_layers = cfg["num_hash_layers"]
        self.vocab_size = cfg["vocab_size"]
        self.hc_mult = cfg["hc_mult"]
        self.mix_hc = (2 + self.hc_mult) * self.hc_mult
        self.compress_ratio = cfg["compress_ratios"][self.layer_num_]
        self.has_compressor = self.compress_ratio != 0
        self.has_indexer = self.compress_ratio == 4
        self.is_hash = self.layer_num_ < self.num_hash_layers
        assert self.n_heads % self.tp_world_size_ == 0
        assert self.o_groups % self.tp_world_size_ == 0
        assert self.index_n_heads % self.tp_world_size_ == 0
        self.prefix = f"layers.{self.layer_num_}"

    def _init_weight_names(self):
        return

    def _init_weight(self):
        self._init_attn()
        if self.has_compressor:
            self._init_compressor(f"{self.prefix}.attn.compressor", self.head_dim, self.compress_ratio)
        if self.has_indexer:
            self._init_indexer()
        self._init_moe()
        self._init_norm()
        self._init_hyper_connection()

    # ------------------------------------------------------------------ attention
    def _init_attn(self):
        p = f"{self.prefix}.attn"
        # q low-rank (a replicated, b column-parallel over heads), kv single head (replicated)
        self.wq_a_ = ROWMMWeight(
            in_dim=self.hidden,
            out_dims=[self.q_lora_rank],
            weight_names=f"{p}.wq_a.weight",
            data_type=self.data_type_,
            quant_method=self.fp8_quant,
            tp_rank=0,
            tp_world_size=1,
        )
        self.wq_b_ = ROWMMWeight(
            in_dim=self.q_lora_rank,
            out_dims=[self.n_heads * self.head_dim],
            weight_names=f"{p}.wq_b.weight",
            data_type=self.data_type_,
            quant_method=self.fp8_quant,
        )
        self.wkv_ = ROWMMWeight(
            in_dim=self.hidden,
            out_dims=[self.head_dim],
            weight_names=f"{p}.wkv.weight",
            data_type=self.data_type_,
            quant_method=self.fp8_quant,
            tp_rank=0,
            tp_world_size=1,
        )
        self.q_norm_ = RMSNormWeight(dim=self.q_lora_rank, weight_name=f"{p}.q_norm.weight", data_type=self.data_type_)
        self.kv_norm_ = RMSNormWeight(dim=self.head_dim, weight_name=f"{p}.kv_norm.weight", data_type=self.data_type_)
        self.attn_sink_ = TpAttSinkWeight(
            all_q_head_num=self.n_heads, weight_name=f"{p}.attn_sink", data_type=torch.float32
        )
        # grouped low-rank output projection: wo_a is a per-group batched matmul [groups, in, o_lora],
        # wo_b is row-parallel [groups*o_lora -> hidden]. wo_a is reshaped in load_hf_weights.
        per_group_in = self.n_heads * self.head_dim // self.o_groups
        self.wo_a_ = ROWBMMWeight(
            dim0=self.o_groups,
            dim1=per_group_in,
            dim2=self.o_lora_rank,
            weight_names=f"{p}.wo_a.weight",
            data_type=self.data_type_,
            quant_method=None,
        )
        self.wo_b_ = COLMMWeight(
            in_dim=self.o_groups * self.o_lora_rank,
            out_dims=[self.hidden],
            weight_names=f"{p}.wo_b.weight",
            data_type=self.data_type_,
            quant_method=self.fp8_quant,
        )

    # ------------------------------------------------------------------ compressor / indexer
    def _init_compressor(self, prefix, head_dim, ratio):
        coff = 2 if ratio == 4 else 1
        # wkv/wgate are bf16 (no scale) and replicated (single KV head).
        self.compressor_wkv_ = ROWMMWeight(
            in_dim=self.hidden,
            out_dims=[coff * head_dim],
            weight_names=f"{prefix}.wkv.weight",
            data_type=self.data_type_,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
        self.compressor_wgate_ = ROWMMWeight(
            in_dim=self.hidden,
            out_dims=[coff * head_dim],
            weight_names=f"{prefix}.wgate.weight",
            data_type=self.data_type_,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
        self.compressor_norm_ = RMSNormWeight(
            dim=head_dim, weight_name=f"{prefix}.norm.weight", data_type=self.data_type_
        )
        self.compressor_ape_ = ParameterWeight(
            weight_name=f"{prefix}.ape", data_type=torch.float32, weight_shape=(ratio, coff * head_dim)
        )

    def _init_indexer(self):
        p = f"{self.prefix}.attn.indexer"
        # wq_b is FP8 in the checkpoint -> de-quantized to bf16 at load; column-parallel over index heads.
        self.idx_wq_b_ = ROWMMWeight(
            in_dim=self.q_lora_rank,
            out_dims=[self.index_n_heads * self.index_head_dim],
            weight_names=f"{p}.wq_b.weight",
            data_type=self.data_type_,
            quant_method=self.fp8_quant,
        )
        self.idx_weights_proj_ = ROWMMWeight(
            in_dim=self.hidden,
            out_dims=[self.index_n_heads],
            weight_names=f"{p}.weights_proj.weight",
            data_type=self.data_type_,
            quant_method=None,
        )
        coff = 2  # indexer compressor always uses ratio 4 (overlap)
        self.idx_cmp_wkv_ = ROWMMWeight(
            in_dim=self.hidden,
            out_dims=[coff * self.index_head_dim],
            weight_names=f"{p}.compressor.wkv.weight",
            data_type=self.data_type_,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
        self.idx_cmp_wgate_ = ROWMMWeight(
            in_dim=self.hidden,
            out_dims=[coff * self.index_head_dim],
            weight_names=f"{p}.compressor.wgate.weight",
            data_type=self.data_type_,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
        self.idx_cmp_norm_ = RMSNormWeight(
            dim=self.index_head_dim, weight_name=f"{p}.compressor.norm.weight", data_type=self.data_type_
        )
        self.idx_cmp_ape_ = ParameterWeight(
            weight_name=f"{p}.compressor.ape", data_type=torch.float32, weight_shape=(4, coff * self.index_head_dim)
        )

    # ------------------------------------------------------------------ moe
    def _init_moe(self):
        p = f"{self.prefix}.ffn"
        # router gate (replicated)
        self.gate_weight_ = ROWMMWeight(
            in_dim=self.hidden,
            out_dims=[self.n_routed_experts],
            weight_names=f"{p}.gate.weight",
            data_type=self.data_type_,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
        if self.is_hash:
            self.gate_tid2eid_ = ParameterWeight(
                weight_name=f"{p}.gate.tid2eid",
                data_type=torch.int64,
                weight_shape=(self.vocab_size, self.network_config_["num_experts_per_tok"]),
            )
        else:
            self.gate_bias_ = ParameterWeight(
                weight_name=f"{p}.gate.bias", data_type=torch.float32, weight_shape=(self.n_routed_experts,)
            )
        # shared expert (dense, bf16 after de-quant): w1=gate, w3=up (row), w2=down (col)
        sp = f"{p}.shared_experts"
        self.shared_gate_ = ROWMMWeight(
            in_dim=self.hidden,
            out_dims=[self.moe_inter],
            weight_names=f"{sp}.w1.weight",
            data_type=self.data_type_,
            quant_method=self.fp8_quant,
        )
        self.shared_up_ = ROWMMWeight(
            in_dim=self.hidden,
            out_dims=[self.moe_inter],
            weight_names=f"{sp}.w3.weight",
            data_type=self.data_type_,
            quant_method=self.fp8_quant,
        )
        self.shared_down_ = COLMMWeight(
            in_dim=self.moe_inter,
            out_dims=[self.hidden],
            weight_names=f"{sp}.w2.weight",
            data_type=self.data_type_,
            quant_method=self.fp8_quant,
        )
        self.experts_ = DeepseekV4FP4ExpertsWeight(
            weight_prefix=f"{p}.experts",
            n_routed_experts=self.n_routed_experts,
            hidden_size=self.hidden,
            moe_intermediate_size=self.moe_inter,
            data_type=self.data_type_,
        )

    def _init_norm(self):
        self.attn_norm_ = RMSNormWeight(
            dim=self.hidden, weight_name=f"{self.prefix}.attn_norm.weight", data_type=self.data_type_
        )
        self.ffn_norm_ = RMSNormWeight(
            dim=self.hidden, weight_name=f"{self.prefix}.ffn_norm.weight", data_type=self.data_type_
        )

    def _init_hyper_connection(self):
        for which in ["attn", "ffn"]:
            setattr(
                self,
                f"hc_{which}_fn_",
                ParameterWeight(
                    weight_name=f"{self.prefix}.hc_{which}_fn",
                    data_type=torch.float32,
                    weight_shape=(self.mix_hc, self.hc_mult * self.hidden),
                ),
            )
            setattr(
                self,
                f"hc_{which}_base_",
                ParameterWeight(
                    weight_name=f"{self.prefix}.hc_{which}_base", data_type=torch.float32, weight_shape=(self.mix_hc,)
                ),
            )
            setattr(
                self,
                f"hc_{which}_scale_",
                ParameterWeight(
                    weight_name=f"{self.prefix}.hc_{which}_scale", data_type=torch.float32, weight_shape=(3,)
                ),
            )

    # ------------------------------------------------------------------ loading
    def load_hf_weights(self, weights):
        self._dequant_in_place(weights)
        return super().load_hf_weights(weights)

    def _direct_fp8_weight_names(self):
        names = set()
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            quant_method = getattr(attr, "quant_method", None)
            if getattr(quant_method, "method_name", None) == "deepgemm-fp8w8a8-b128":
                names.update(getattr(attr, "weight_names", []))
        return names

    def _dequant_in_place(self, weights):
        p = self.prefix + "."
        direct_fp8_names = self._direct_fp8_weight_names()
        # Convert every (weight, scale) pair belonging to this layer. Existing FP8 matmul
        # weights stay quantized; bmm-only weights are expanded; routed FP4 experts stay packed.
        for k in [k for k in list(weights.keys()) if k.startswith(p) and k.endswith(".weight")]:
            scale_k = k[: -len(".weight")] + ".scale"
            if scale_k not in weights:
                continue
            w, s = weights[k], weights[scale_k]
            if w.dtype == torch.int8:  # FP4 routed experts stay packed for DeepseekV4FP4ExpertsWeight.
                continue
            elif k in direct_fp8_names:  # FP8 e4m3, block-128 scale, run by DeepGEMM directly
                weights[k.replace("weight", "weight_scale_inv")] = s.to(torch.float32)
                del weights[scale_k]
            else:  # FP8 e4m3 for no-quant paths such as ROWBMMWeight
                weights[k] = dequant_fp8_block_to_bf16(w, s).to(self.data_type_)
                del weights[scale_k]
        # grouped-O: reshape [groups*o_lora, in] -> [groups, in, o_lora] for the batched matmul
        woa = f"{self.prefix}.attn.wo_a.weight"
        if woa in weights and weights[woa].dim() == 2:
            w = weights[woa]
            per_group_in = self.n_heads * self.head_dim // self.o_groups
            weights[woa] = w.view(self.o_groups, self.o_lora_rank, per_group_in).transpose(1, 2).contiguous()
        return
