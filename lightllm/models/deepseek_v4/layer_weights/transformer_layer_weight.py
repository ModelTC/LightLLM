import torch
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    ROWMMWeight,
    COLMMWeight,
    ROWBMMWeight,
    RMSNormWeight,
    ParameterWeight,
    TpAttSinkWeight,
    FusedMoeWeight,
)
from ..triton_kernel.quant_convert import dequant_fp8_block_to_bf16


class DeepseekV4TransformerLayerWeight(TransformerLayerWeight):
    """Per-layer weights for DeepSeek-V4-Flash.

    DS4 does not share DS2/DS3.2's ``model.layers.*.self_attn/mlp`` layout. Its attention is
    HC + CSA, and routed experts are checkpointed as MXFP4 (fp4 release) or
    FP8 block-128 (fp8 release, same layout as the dense fp8 weights).
    """

    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    def _parse_config(self):
        cfg = self.network_config_
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
        self.prefix = f"layers.{self.layer_num_}"

    def _init_weight(self):
        self._init_qkvo()
        if self.has_compressor:
            self._init_compressor()
        if self.has_indexer:
            self._init_indexer()
        self._init_moe()
        self._init_norm()
        self._init_hyper_connection()

    # ------------------------------------------------------------------ attention
    def _init_qkvo(self):
        p = f"{self.prefix}.attn"
        # q low-rank (a replicated, b column-parallel over heads), kv single head (replicated)
        self.wq_a_ = ROWMMWeight(
            in_dim=self.hidden,
            out_dims=[self.q_lora_rank],
            weight_names=f"{p}.wq_a.weight",
            data_type=self.data_type_,
            quant_method=self.get_quant_method("wq_a"),
            tp_rank=0,
            tp_world_size=1,
        )
        self.wq_b_ = ROWMMWeight(
            in_dim=self.q_lora_rank,
            out_dims=[self.n_heads * self.head_dim],
            weight_names=f"{p}.wq_b.weight",
            data_type=self.data_type_,
            quant_method=self.get_quant_method("wq_b"),
        )
        self.wkv_ = ROWMMWeight(
            in_dim=self.hidden,
            out_dims=[self.head_dim],
            weight_names=f"{p}.wkv.weight",
            data_type=self.data_type_,
            quant_method=self.get_quant_method("wkv"),
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
            quant_method=self.get_quant_method("wo_b"),
        )

    # ------------------------------------------------------------------ compressor / indexer
    def _init_compressor(self):
        prefix = f"{self.prefix}.attn.compressor"
        head_dim = self.head_dim
        ratio = self.compress_ratio

        coff = 2 if ratio == 4 else 1
        # wkv/wgate are bf16 (no scale) and replicated (single KV head).
        self.compressor_wkv_gate_ = ROWMMWeight(
            in_dim=self.hidden,
            out_dims=[coff * head_dim, coff * head_dim],
            weight_names=[f"{prefix}.wkv.weight", f"{prefix}.wgate.weight"],
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
        # The Lightning-Indexer is REPLICATED across TP ranks (like sglang/vllm), not head-sharded:
        # q_lora and the attn input are already full on every rank, so each rank scores all
        # index_n_heads locally and the c4 top-k is identical everywhere -- no gather/all_reduce.
        # wq_b is FP8 in the checkpoint -> de-quantized to bf16 at load.
        self.idx_wq_b_ = ROWMMWeight(
            in_dim=self.q_lora_rank,
            out_dims=[self.index_n_heads * self.index_head_dim],
            weight_names=f"{p}.wq_b.weight",
            data_type=self.data_type_,
            quant_method=self.get_quant_method("idx_wq_b"),
            tp_rank=0,
            tp_world_size=1,
        )
        self.idx_weights_proj_ = ROWMMWeight(
            in_dim=self.hidden,
            out_dims=[self.index_n_heads],
            weight_names=f"{p}.weights_proj.weight",
            data_type=self.data_type_,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
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
        # Router gate in bf16 (matches the sglang/vLLM DeepSeek references, which run the gate GEMM in
        # the model dtype); the bf16 GEMM output is cast back to fp32 in _ffn for topk_hash_softplus_sqrt.
        self.gate_weight_ = ROWMMWeight(
            in_dim=self.hidden,
            out_dims=[self.n_routed_experts],
            weight_names=f"{p}.gate.weight",
            data_type=torch.bfloat16,
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
        # shared expert (dense, bf16 after de-quant): w1=gate, w3=up fused (row), w2=down (col).
        # Named gate_up_proj/down_proj so the inherited Llama `_ffn_tp` (fused gate_up matmul +
        # silu_and_mul triton kernel, no swiglu clamp) drives it directly. Order [w1, w3] = [gate, up]
        # matches silu_and_mul_fwd's blocked layout (first half gate, second half up).
        sp = f"{p}.shared_experts"
        self.gate_up_proj = ROWMMWeight(
            in_dim=self.hidden,
            out_dims=[self.moe_inter, self.moe_inter],
            weight_names=[f"{sp}.w1.weight", f"{sp}.w3.weight"],
            data_type=self.data_type_,
            quant_method=self.get_quant_method("shared_gate"),
        )
        self.down_proj = COLMMWeight(
            in_dim=self.moe_inter,
            out_dims=[self.hidden],
            weight_names=f"{sp}.w2.weight",
            data_type=self.data_type_,
            quant_method=self.get_quant_method("shared_down"),
        )
        self.experts_ = FusedMoeWeight(
            gate_proj_name="w1",
            down_proj_name="w2",
            up_proj_name="w3",
            e_score_correction_bias_name="",
            weight_prefix=f"{p}.experts",
            n_routed_experts=self.n_routed_experts,
            hidden_size=self.hidden,
            moe_intermediate_size=self.moe_inter,
            data_type=self.data_type_,
            quant_method=self.quant_cfg.get_quant_method(self.layer_num_, "fused_moe"),
            layer_num=self.layer_num_,
            network_config=self.network_config_,
        )

    def _init_norm(self):
        self.attn_norm_ = RMSNormWeight(
            dim=self.hidden, weight_name=f"{self.prefix}.attn_norm.weight", data_type=self.data_type_
        )
        self.ffn_norm_ = RMSNormWeight(
            dim=self.hidden, weight_name=f"{self.prefix}.ffn_norm.weight", data_type=self.data_type_
        )

    def _init_hyper_connection(self):
        p = self.prefix
        self.hc_attn_fn_ = ParameterWeight(
            weight_name=f"{p}.hc_attn_fn",
            data_type=torch.float32,
            weight_shape=(self.mix_hc, self.hc_mult * self.hidden),
        )
        self.hc_attn_base_ = ParameterWeight(
            weight_name=f"{p}.hc_attn_base", data_type=torch.float32, weight_shape=(self.mix_hc,)
        )
        self.hc_attn_scale_ = ParameterWeight(
            weight_name=f"{p}.hc_attn_scale", data_type=torch.float32, weight_shape=(3,)
        )
        self.hc_ffn_fn_ = ParameterWeight(
            weight_name=f"{p}.hc_ffn_fn",
            data_type=torch.float32,
            weight_shape=(self.mix_hc, self.hc_mult * self.hidden),
        )
        self.hc_ffn_base_ = ParameterWeight(
            weight_name=f"{p}.hc_ffn_base", data_type=torch.float32, weight_shape=(self.mix_hc,)
        )
        self.hc_ffn_scale_ = ParameterWeight(
            weight_name=f"{p}.hc_ffn_scale", data_type=torch.float32, weight_shape=(3,)
        )

    # ------------------------------------------------------------------ loading
    def load_hf_weights(self, weights):
        self._dequant_in_place(weights)
        return super().load_hf_weights(weights)

    def _fp8_scale_renames(self):
        """Map weight name -> the scale name its quant method loads (e.g. `weight_scale_inv`
        for DeepGEMM). Read from each MM weight's own `weight_scale_names`, so the rename
        target always matches what that weight will look up; no-quant weights have None
        entries and are skipped."""
        renames = {}
        for attr in self.__dict__.values():
            weight_names = getattr(attr, "weight_names", ())
            scale_names = getattr(attr, "weight_scale_names", ())
            for weight_name, scale_name in zip(weight_names, scale_names):
                if scale_name is not None:
                    renames[weight_name] = scale_name
        return renames

    def _dequant_in_place(self, weights):
        p = self.prefix + "."
        scale_renames = self._fp8_scale_renames()
        # Convert every `.scale` belonging to this layer. Weights are loaded incrementally
        # per safetensors shard, so the paired weight may live in another shard:
        # - routed expert `.scale` follows the fused_moe quant method's weight_scale_suffix:
        #   MXFP4 consumes `.scale` as-is, FP8 DeepGEMM expects `.weight_scale_inv` (rename only);
        # - FP8 matmul scales only need renaming for DeepGEMM, no weight required;
        # - FP8 pairs on no-quant paths (wo_a's ROWBMMWeight) are expanded to bf16,
        #   the only case that truly requires weight and scale in the same shard.
        expert_scale_suffix = self.experts_.quant_method.weight_scale_suffix
        for scale_k in [k for k in list(weights.keys()) if k.startswith(p) and k.endswith(".scale")]:
            if scale_k.startswith(f"{p}ffn.experts."):
                if expert_scale_suffix is not None and expert_scale_suffix != "scale":
                    weights[scale_k[: -len("scale")] + expert_scale_suffix] = weights[scale_k].to(torch.float32)
                    del weights[scale_k]
                continue
            k = scale_k[: -len(".scale")] + ".weight"
            target = scale_renames.get(k)
            if target is not None:  # FP8 e4m3, block-128 scale, run by DeepGEMM directly
                weights[target] = weights[scale_k].to(torch.float32)
                del weights[scale_k]
            else:
                weights[k] = dequant_fp8_block_to_bf16(weights[k], weights[scale_k]).to(self.data_type_)
                del weights[scale_k]
        # grouped-O: reshape [groups*o_lora, in] -> [groups, in, o_lora] for the batched matmul
        woa = f"{self.prefix}.attn.wo_a.weight"
        if woa in weights and weights[woa].dim() == 2:
            w = weights[woa]
            per_group_in = self.n_heads * self.head_dim // self.o_groups
            weights[woa] = w.view(self.o_groups, self.o_lora_rank, per_group_in).transpose(1, 2).contiguous()
        return
