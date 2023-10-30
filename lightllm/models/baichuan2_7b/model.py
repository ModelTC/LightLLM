import os
import json
import torch

from lightllm.models.baichuan2_7b.layer_weights.pre_and_post_layer_weight import Baichuan2_7bPreAndPostLayerWeight
from lightllm.models.baichuan7b.model import Baichuan7bTpPartModel
from lightllm.models.baichuan2_7b.layer_infer.transformer_layer_infer import Baichuan2_7bTransformerLayerInfer


class Baichuan2_7bTpPartModel(Baichuan7bTpPartModel):
    # weight class
    pre_and_post_weight_class = Baichuan2_7bPreAndPostLayerWeight

    # infer class
    transformer_layer_infer_class = Baichuan2_7bTransformerLayerInfer

    def __init__(self, tp_rank, world_size, weight_dir, max_total_token_num, load_way="HF", mode=[]):
        super().__init__(tp_rank, world_size, weight_dir, max_total_token_num, load_way, mode)

    def _init_to_get_rotary(self, default_base=10000.0):
        if self.config.get("rope_scaling", {}) is None:
            rope_scaling_factor = 1.0
        else:
            rope_scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)

        base = self.config.get("rope_theta", float(default_base))

        if "max_sequence_length" in self.config:
            max_seq_len = self.config["max_sequence_length"]
        else:
            max_position_embeddings = self.config.get(
                "max_position_embeddings",
                2048 if base <= 10000.0 + 1e-5 else 16384
            )
            max_seq_len = max_position_embeddings * rope_scaling_factor

        # NTK
        try:
            ntk_alpha = float(os.environ.get("LIGHTLLM_NTK_ALPHA", 1))
            assert ntk_alpha >= 1
            if ntk_alpha > 1:
                print(f"Note: NTK enabled, alpha set to {ntk_alpha}")
            max_seq_len *= ntk_alpha
            base = base * (ntk_alpha ** (self.head_dim_ / (self.head_dim_-2))) #Base change formula
        except:
            pass

        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32) / self.head_dim_))
        # t = torch.arange(max_seq_len + 1024 * 64, device="cpu", dtype=torch.float32) / rope_scaling_factor
        t = torch.arange(max_seq_len, device="cpu", dtype=torch.float32) / rope_scaling_factor
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(torch.float32).cuda()
        self._sin_cached = torch.sin(freqs).to(torch.float32).cuda()
        return
