from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from lightllm.common.basemodel.layer_weights import hf_load_utils
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight
from lightllm.models.deepseek_v41.model import DeepseekV41TpPartModel


def test_ep_loader_skips_nonlocal_tensor_reads_and_keeps_eplb_replicas(tmp_path, monkeypatch):
    owner = FusedMoeWeight.__new__(FusedMoeWeight)
    owner.weight_prefix = "layers.0.ffn.experts"
    owner.expert_idx_to_local_idx = {1: 0}
    owner._initial_redundant_expert_idx_to_local_idx = {3: 1}
    model = DeepseekV41TpPartModel.__new__(DeepseekV41TpPartModel)
    model.args = SimpleNamespace(enable_ep_moe=True)
    model.trans_layers_weight = [SimpleNamespace(experts=owner)]
    weights = {
        f"{owner.weight_prefix}.{expert}.w1.{suffix}": torch.full((2, 2), expert)
        for expert in range(4)
        for suffix in ("weight", "scale")
    }
    weights["layers.0.ffn.shared_experts.w1.weight"] = torch.ones(2, 2)
    weights["layers.0.engram.embed.weight"] = torch.zeros(2, 2)
    save_file(weights, tmp_path / "model.safetensors")
    original = hf_load_utils.safe_open
    reads, loaded = [], {}

    class Checkpoint:
        def __init__(self, *args):
            self.source = original(*args)

        def __enter__(self):
            self.source.__enter__()
            return self

        def __exit__(self, *args):
            return self.source.__exit__(*args)

        def keys(self):
            return self.source.keys()

        def get_tensor(self, key):
            reads.append(key)
            return self.source.get_tensor(key)

    monkeypatch.setattr(hf_load_utils, "safe_open", Checkpoint)
    monkeypatch.setattr(hf_load_utils, "get_current_device_id", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    hf_load_utils.load_func(
        "model.safetensors",
        use_safetensors=True,
        weight_dir=tmp_path,
        pre_post_layer=SimpleNamespace(load_hf_weights=loaded.update),
        weight_key_filter=model._filter_hf_weight_keys,
    )
    expected = {key for key in weights if ".experts.0." not in key and ".experts.2." not in key}
    assert set(reads) == expected
    assert set(loaded) == expected
    for key in expected:
        torch.testing.assert_close(loaded[key], weights[key])
    model.args.enable_ep_moe = False
    assert model._filter_hf_weight_keys(list(weights)) == list(weights)
