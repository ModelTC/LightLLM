import torch
from typing import Callable, Dict

from lightllm.common.basemodel import TransformerLayerWeight
from .qlinear import MixedQLinear, SharedQuantizedInput

class LlamaTransformerLayerWeightQuik(TransformerLayerWeight):
    WEIGHT_KEYS = [
        "fp_weight", # fp16 weights for outliers, which length is 256 or 768 usually
        "int_weight", # symetric per-channel quantized weights, int8 for down_proj and int4 (packed to int8) for other proj
        "weights_scales", # symetric per-channel quantized scales, dtype is fp16
        "reduced_w", # reduced_w used to dequantize
    ]
    INDICE_KEYS = [
        "fp_indices", # int64 indices of outliers
        "int_indices", # int64 indices of quantized weights
    ]
    ALL_WEIGHT_KEYS = WEIGHT_KEYS + INDICE_KEYS

    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        self.cat_kv_ = True
        self.cat_gate_up_ = True

        self._q_weights: Dict[str, torch.Tensor] = {}
        self._k_weights: Dict[str, torch.Tensor] = {}
        self._v_weights: Dict[str, torch.Tensor] = {}
        self._o_weights: Dict[str, torch.Tensor] = {}

        self._gate_weights: Dict[str, torch.Tensor] = {}
        self._up_weights: Dict[str, torch.Tensor] = {}
        self._down_weights: Dict[str, torch.Tensor] = {}

        self.q_proj: MixedQLinear = None
        self.k_proj: MixedQLinear = None
        self.v_proj: MixedQLinear = None
        self.kv_proj: MixedQLinear = None
        self.gate_proj: MixedQLinear = None
        self.up_proj: MixedQLinear = None
        self.gate_up_proj: MixedQLinear = None
        self.down_up_proj: MixedQLinear = None

    def load_hf_weights(self, weights):
        self._load_qkvo_weights(weights)
        self._load_ffn_weights(weights)

    def verify_load(self):
        errors = "weights load not ok"
        norm_weights = [
            self.att_norm_weight_,
            self.ffn_norm_weight_,
        ]
        proj_weights = [
            self._q_weights,
            self._k_weights,
            self._v_weights,
            self._o_weights,
            self._gate_weights,
            self._up_weights,
            self._down_weights,
        ]
        weights = norm_weights + proj_weights
        for i in range(len(weights)):
            assert weights[i] is not None, f"index: {i} {errors}"
            if isinstance(weights[i], dict):
                expect = set(self.ALL_WEIGHT_KEYS)
                current = set(weights[i].keys())

                missings = expect - current
                unexpected = current - expect
                assert len(missings) == 0 and len(unexpected) == 0, f"index: {i} weights load not ok, missings = {missings}, unexpected = {unexpected}"

        # NOTE load projs here to avoid override model._init_weights()
        self._load_projs()
        self._verify_proj()

        return

    def _verify_proj(self):
        errors = "proj load not ok"
        kv_proj = [self.kv_proj] if self.cat_kv_ else [self.k_proj, self.v_proj]
        gate_up_proj = [self.gate_up_proj] if self.cat_gate_up_ else [self.gate_proj, self.up_proj]
        weights = [
            self.q_proj,
            self.o_proj,
            self.down_proj,
        ] + kv_proj + gate_up_proj
        for i in range(len(weights)):
            assert weights[i] is not None, f"index: {i} {errors}"
        return

    def _load_projs(self):
        self._load_qkvo_proj()
        self._load_ffn_proj()

    # @TODO: 测一下QKV都合并
    def _try_cat_tensors(self, first_weights: Dict[str, torch.Tensor], second_weights: Dict[str, torch.Tensor], handle_func=None):
        """Q/K/V and Gate/Up projection has the same outlier indices, so K and V can be concatenated."""
        assert len(first_weights) == len(second_weights), "first and second weights len dismatch"
        assert all([(first_weights[k].shape == second_weights[k].shape and first_weights[k].dtype == first_weights[k].dtype and first_weights[k].device == first_weights[k].device) for k in self.ALL_WEIGHT_KEYS]), "first and second weights dismatch"

        with self.lock:
            kv_weights = {}
            for key in self.WEIGHT_KEYS:
                cat_dim = 1 if key == "reduced_w" else 0
                # weights shape: [out, in], should concat by out axis
                kv_weights[key] = torch.cat([first_weights[key], second_weights[key]], dim=cat_dim).contiguous()
                if handle_func != None and isinstance(handle_func, Callable):
                    kv_weights[key] = handle_func(kv_weights[key])

            for key in self.INDICE_KEYS:
                kv_weights[key] = first_weights[key]
            return kv_weights

    def _load_weights(self, prefix: str, weights: Dict[str, torch.Tensor]):
        """load the weights of projection

        Args:
            prefix (str): The weight prefix in the weights dictionary. 
                Example: "model.layers.0.self_attn.q_proj", "model.layers.0.mlp.gate_proj".
            weights (Dict[str, torch.Tensor]): The weights dictionary, reduced_w's shape is [1, N] and other weights' shape are [N, K].

        Returns:
            Dict[str, torch.Tensor]
        """
        _construt_weight_key_fn = lambda x: f"{prefix}.{x}"
        collector = {}

        # load weights
        for name in self.ALL_WEIGHT_KEYS:
            full_name = _construt_weight_key_fn(name)
            if full_name not in weights:
                continue
            collector[name] = weights[full_name]

        return collector


    def _split_weights_by_column(self, weights: Dict[str, torch.Tensor], split_n: int):
        """split q/k/v_proj in attention module and gate/up_proj in mlp module, which are column splitting when tensor parallel is enabled

        Args:
                Example: "model.layers.0.self_attn.q_proj", "model.layers.0.mlp.gate_proj".
            split_n (int): The number of splits.
            weights (Dict[str, torch.Tensor]): The weights dictionary, reduced_w's shape is [1, N] and other weights' shape are [N, K].

        Returns:
            Dict[str, torch.Tensor]
        """
        if self.world_size_ == 1:
            return weights

        collector = {}

        # load weights
        for name in self.WEIGHT_KEYS:
            weight_ = weights[name]
            if name == "reduced_w":
                weight_ = weight_[:, split_n * self.tp_rank_ : split_n * (self.tp_rank_ + 1)]
            else:
                weight_ = weight_[split_n * self.tp_rank_ : split_n * (self.tp_rank_ + 1), :]
            collector[name] = weight_

        # load indices
        for name in self.INDICE_KEYS:
            weight_ = weights[name]
            # weight_ = weight_.to(torch.int32)
            collector[name] = weight_

        return collector

    def _split_weights_by_row(self, weights: Dict[str, torch.Tensor], split_n: int):
        """split o_proj in attention module and down_proj in mlp module, which are row splitting when tensor parallel is enabled

        Args:
            split_n (int): The number of splits.
            weights (Dict[str, torch.Tensor]): The weights dictionary, reduced_w's shape is [1, N] and other weights' shape are [N, K].

        Returns:
            Dict[str, torch.Tensor]
        """
        if self.world_size_ == 1:
            return weights

        collector = {}

        lower = split_n * self.tp_rank_
        upper = split_n * (self.tp_rank_ + 1)

        fp_indices_num = weights["fp_indices"].shape[0]
        int_indices_num = weights["int_indices"].shape[0]

        N = weights["int_weight"].shape[0]
        K = fp_indices_num + int_indices_num

        bits = weights["int_weight"].shape[1] * 8 // int_indices_num
        assert bits in [4, 8], f"quant bits should be one of [4, 8], but get {bits}"

        # load indices
        weight_ = weights["fp_indices"]
        # only indices in range [lower, upper) are reserved
        fp_selected = torch.logical_and(weight_ >= lower, weight_ < upper)
        collector["fp_indices"] = weight_[fp_selected]

        weight_ = weights["int_indices"]
        # only indices in range [lower, upper) are reserved
        int_selected = torch.logical_and(weight_ >= lower, weight_ < upper)
        collector["int_indices"] = weight_[int_selected]

        # load weights
        # import pdb; pdb.set_trace()
        weight_ = weights["fp_weight"]
        collector["fp_weight"] = weight_[:, fp_selected]

        weight_ = weights["int_weight"]
        if bits == 8:
            collector["int_weight"] = weight_[:, int_selected]
            dequant_weight_ = collector["int_weight"] * weights["weights_scales"]
            collector["reduced_w"] = torch.sum(dequant_weight_, dim = 1, dtype = torch.float16).unsqueeze(0)
        else:
            in_dim, out_dim = weight_.shape
            out_dim = out_dim * 2 # uint8 unpack to [int4, int4]
            assert int_selected.shape[0] % 2 == 0, f"TP is {self.world_size_}, total indices is {out_dim}. rank[{self.tp_rank_}] indices is {int_selected.shape[0]}"

            unpack = weight_.flatten().unsqueeze(1)
            unpack = unpack.repeat([1, 2])
            unpack[:, 0] = unpack[:, 0] & 0xf
            unpack[:, 1] = (unpack[:, 1] >> 4) & 0xf
            unpack = unpack.view(in_dim, out_dim)
            weight_ = unpack[:, int_selected]
            # reinterpret int4 weight by bits
            int4_weight_ = (weight_.to(torch.int8) << 4) >> 4
            dequant_weight_ = int4_weight_ * weights["weights_scales"]
            collector["reduced_w"] = torch.sum(dequant_weight_, dim = 1, dtype = torch.float16).unsqueeze(0)
            # repack two int4 to uint8
            collector["int_weight"] = weight_[:, 0::2] | (weight_[:, 1::2] << 4)

        collector["weights_scales"] = weights["weights_scales"]

        for name in self.INDICE_KEYS:
            collector[name] = collector[name] - lower

        return collector

    def _load_qkvo_weights(self, weights):
        # input layernorm params
        if f"model.layers.{self.layer_num_}.input_layernorm.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.input_layernorm.weight"])

        q_weights_ = self._load_weights(f"model.layers.{self.layer_num_}.self_attn.q_proj", weights)
        k_weights_ = self._load_weights(f"model.layers.{self.layer_num_}.self_attn.k_proj", weights)
        v_weights_ = self._load_weights(f"model.layers.{self.layer_num_}.self_attn.v_proj", weights)
        o_weights_ = self._load_weights(f"model.layers.{self.layer_num_}.self_attn.o_proj", weights)
        with self.lock:
            self._q_weights.update(q_weights_)
            self._k_weights.update(k_weights_)
            self._v_weights.update(v_weights_)
            self._o_weights.update(o_weights_)

        return

    def _load_ffn_weights(self, weights):
        if f"model.layers.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(
                weights[f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"]
            )

        gate_weights_ = self._load_weights(f"model.layers.{self.layer_num_}.mlp.gate_proj", weights)
        up_weights_ = self._load_weights(f"model.layers.{self.layer_num_}.mlp.up_proj", weights)
        down_weights_ = self._load_weights(f"model.layers.{self.layer_num_}.mlp.down_proj", weights)

        with self.lock:
            self._gate_weights.update(gate_weights_)
            self._up_weights.update(up_weights_)
            self._down_weights.update(down_weights_)

        return

    def _load_proj(self, name_hint:str, weights: Dict[str, torch.Tensor], shared_input: SharedQuantizedInput = None) -> MixedQLinear:
        if not weights:
            return None
        try:
            proj = MixedQLinear.from_dict(name_hint, weights, shared_input)
            if self.tp_rank_ is None:
                return proj.eval().cuda()
            else:
                return proj.eval().cuda(self.tp_rank_)
        except Exception as e:
            print(str(e))
            return None

    def _load_qkvo_proj(self):
        n_embed = self.network_config_["hidden_size"]
        q_split_n_embed = n_embed // self.world_size_
        kv_split_n_embed = (
            n_embed
            // self.network_config_["num_attention_heads"]
            * self.network_config_["num_key_value_heads"]
            // self.world_size_
        )

        self._q_weights = self._split_weights_by_column(self._q_weights, q_split_n_embed)
        self._k_weights = self._split_weights_by_column(self._k_weights, kv_split_n_embed)
        self._v_weights = self._split_weights_by_column(self._v_weights, kv_split_n_embed)
        self._o_weights = self._split_weights_by_row(self._o_weights, q_split_n_embed)

        shared_input_n = 2 if self.cat_kv_ else 3
        qkv_shared_input = SharedQuantizedInput(shared_input_n)
        self.q_proj = self._load_proj(f"model.layers.{self.layer_num_}.self_attn.q_proj", self._q_weights, qkv_shared_input)
        self._q_weights = None
        # cat k and v
        if self.cat_kv_:
            kv_weights = self._try_cat_tensors(self._k_weights, self._v_weights)
            self.k_weights_ = None
            self.v_weights_ = None
            self.kv_proj: MixedQLinear = self._load_proj(f"model.layers.{self.layer_num_}.self_attn.kv_proj", kv_weights, qkv_shared_input)
            kv_weights = None
        else: # split k and v
            self.k_proj: MixedQLinear = self._load_proj(f"model.layers.{self.layer_num_}.self_attn.k_proj", self._k_weights, qkv_shared_input)
            self.v_proj: MixedQLinear = self._load_proj(f"model.layers.{self.layer_num_}.self_attn.v_proj", self._v_weights, qkv_shared_input)
            self._k_weights = None
            self._v_weights = None

        self.o_proj: MixedQLinear = self._load_proj(f"model.layers.{self.layer_num_}.self_attn.o_proj", self._o_weights)
        self._o_weights = None

        return

    def _load_ffn_proj(self):
        inter_size = self.network_config_["intermediate_size"]
        split_inter_size = inter_size // self.world_size_

        self._gate_weights = self._split_weights_by_column(self._gate_weights, split_inter_size)
        self._up_weights = self._split_weights_by_column(self._up_weights, split_inter_size)
        self._down_weights = self._split_weights_by_row(self._down_weights, split_inter_size)

        shared_input_n = 1 if self.cat_gate_up_ else 2
        gate_up_shared_input = SharedQuantizedInput(shared_input_n)
        if self.cat_gate_up_:
            # [gate, up]
            gate_up_weights = self._try_cat_tensors(self._gate_weights, self._up_weights)
            self._up_weights = None
            self._gate_weights = None
            self.gate_up_proj = self._load_proj(f"model.layers.{self.layer_num_}.mlp.gate_up", gate_up_weights, gate_up_shared_input)
            gate_up_weights = None
        else:
            self.up_proj: MixedQLinear = self._load_proj(f"model.layers.{self.layer_num_}.mlp.up_proj", self._up_weights, gate_up_shared_input)
            self.gate_proj: MixedQLinear = self._load_proj(f"model.layers.{self.layer_num_}.mlp.gate_proj", self._gate_weights, gate_up_shared_input)

        self.down_proj: MixedQLinear = self._load_proj(f"model.layers.{self.layer_num_}.mlp.down_proj", self._down_weights)

        return
