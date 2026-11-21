from types import SimpleNamespace

import pytest
import torch

from lightllm.models.sensenova_loop.config import get_loop_layer_indices, get_loop_layer_types

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Qwen3.5 layers keep an mrope tensor on CUDA")


@pytest.mark.parametrize(
    "settings",
    [
        {"start": 1, "end": 6, "eval_count": 2, "use_gate": False},
        {"start": 4, "end": 28, "eval_count": 2, "use_gate": False},
    ],
    ids=["unaligned-loop", "aligned-loop"],
)
def test_every_linear_execution_layer_can_split_its_gdn_projection(settings):
    """Execution layers take their kind from the looped layer_types, not from full_attention_interval.

    An unaligned loop puts linear layers where (layer + 1) % full_attention_interval == 0; they still need the
    per-rank GDN dims that _gdn_att_dict uses to split the fused projection.
    """
    from lightllm.models.sensenova_loop.layer_infer.transformer_layer_infer import SenseNovaLoopTransformerLayerInfer

    text = {
        "num_hidden_layers": 32,
        "layer_types": ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(32)],
    }
    kinds = get_loop_layer_types(settings, text)
    config = dict(
        num_hidden_layers=len(kinds),
        layer_types=list(kinds),
        loop_physical_indices=list(get_loop_layer_indices(settings, 32)),
        full_attention_interval=4,
        hidden_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        rms_norm_eps=1e-6,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=48,
    )
    linear_layers = 0
    for layer_num, kind in enumerate(kinds):
        layer = SenseNovaLoopTransformerLayerInfer(layer_num, config)
        assert layer.is_linear_attention_layer == (kind == "linear_attention")
        if not layer.is_linear_attention_layer:
            continue
        linear_layers += 1
        dims = {
            name: getattr(layer, name, None)
            for name in ("tp_linear_num_v_heads", "linear_head_v_dim", "tp_linear_key_dim")
        }
        assert dims == {"tp_linear_num_v_heads": 4, "linear_head_v_dim": 48, "tp_linear_key_dim": 64}, layer_num
        # The layer splits the fused projection for the GDN adapter with these dims.
        qkv, value = 2 * 64 + 4 * 48, 4 * 48
        mixed = torch.arange(3 * (qkv + 2 * value + 2 * 4), dtype=torch.float32).view(3, -1)
        state = SimpleNamespace(get_layer_view=lambda logical_layer_id: ("conv", "ssm"))
        infer_state = SimpleNamespace(
            att_state_role="main", att_state_manager=SimpleNamespace(get_state=lambda **kwargs: state)
        )
        weight = SimpleNamespace(
            linear_conv1d=SimpleNamespace(mm_param=SimpleNamespace(weight=None), bias=None),
            linear_A_log=SimpleNamespace(weight=None),
            linear_dt_bias=SimpleNamespace(weight=None),
        )
        params = layer._gdn_att_dict(mixed, infer_state, weight)
        assert torch.equal(params["mixed_qkv"], mixed[:, :qkv])
        assert torch.equal(params["z"].reshape(3, -1), mixed[:, qkv : qkv + value])
        assert torch.equal(params["b"], mixed[:, qkv + value : qkv + value + 4])
        assert torch.equal(params["a"], mixed[:, qkv + value + 4 :])
    assert linear_layers == sum(kind == "linear_attention" for kind in kinds)
