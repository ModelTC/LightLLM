from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.hidden_collector import NoopHiddenCollector
from lightllm.models.qwen4_exp.layer_infer import replicated_decoder
from lightllm.models.qwen4_exp import model as model_module
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.fixture
def decode_model(monkeypatch):
    def build(world=4, rank=0, mtp=3):
        args = StartArgs()
        args.dp, args.nnodes = 1, 1
        args.llm_kv_type = "fp8kv_spt"
        args.enable_ep_moe = args.enable_tpsp_mix_mode = True
        args.mtp_mode, args.mtp_step = "eagle_with_att", mtp
        args.ssm_state_mode, args.linear_att_ssm_data_type = "replay", "float32"
        args.ssm_replay_config = {"cache_len": 16, "compute_mode": "precompute"}
        count = 512 // world
        none = SimpleNamespace(method_name="none")
        experts = SimpleNamespace(
            quant_method=none,
            local_logic_expert_ids_list=list(range(rank * count, (rank + 1) * count)),
            num_fused_shared_experts=0,
            redundancy_expert_num=0,
            scoring_func="softmax",
            routed_scaling_factor=1.0,
            per_expert_scale=None,
            e_score_correction_bias=None,
            w13=SimpleNamespace(weight=torch.empty(count, 1280, 2560, dtype=torch.bfloat16, device="meta")),
            w2=SimpleNamespace(weight=torch.empty(count, 2560, 640, dtype=torch.bfloat16, device="meta")),
        )
        model = model_module.Qwen4ExpTpPartModel.__new__(model_module.Qwen4ExpTpPartModel)
        model.__dict__.update(
            args=args,
            tp_world_size_=world,
            tp_rank_=rank,
            data_type=torch.bfloat16,
            quant_cfg=SimpleNamespace(quant_type="none", quantized_weight=False, quant_cfg={}),
            config=dict(replicated_decoder._REPLICATED_DECODE_CONFIG),
            is_mtp_draft_model=False,
            hidden_collector_prototype=NoopHiddenCollector(),
            trans_layers_weight=[
                SimpleNamespace(
                    experts=experts,
                    gate_up_proj=SimpleNamespace(quant_method=none),
                    down_proj=SimpleNamespace(quant_method=none),
                    is_linear_attention_layer=True,
                )
            ],
        )
        symmetric = SimpleNamespace(disabled=False, dtype=torch.bfloat16, world_size=world, max_size=32 * 1024 ** 2)
        monkeypatch.setattr(
            replicated_decoder.dist_group_manager, "get_group", lambda _: SimpleNamespace(symm_mem_reduce=symmetric)
        )
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
        return model, symmetric

    return build


@pytest.mark.parametrize("world,rank", [(4, 0), (4, 3), (8, 0), (8, 7)])
def test_replicated_decode_uses_local_expert_shard(decode_model, world, rank):
    model, _ = decode_model(world, rank)
    assert replicated_decoder.supports_replicated_decode(model)
    model.trans_layers_weight[0].experts.local_logic_expert_ids_list.reverse()
    assert not replicated_decoder.supports_replicated_decode(model)


@pytest.mark.parametrize("world", [4, 8])
def test_replicated_decode_rejects_incompatible_weights(decode_model, world):
    model, _ = decode_model(world)
    model.trans_layers_weight[0].experts.w13.weight = torch.empty(
        512 // world, 1280, 2560, dtype=torch.float32, device="meta"
    )
    assert not replicated_decoder.supports_replicated_decode(model)
    model.trans_layers_weight[0].experts.w13.weight = torch.empty(
        512 // world + 1, 1280, 2560, dtype=torch.bfloat16, device="meta"
    )
    assert not replicated_decoder.supports_replicated_decode(model)


@pytest.mark.parametrize("world", [4, 8])
def test_replicated_decode_requires_matching_collective(decode_model, world):
    model, symmetric = decode_model(world)
    symmetric.world_size = 2
    assert not replicated_decoder.supports_replicated_decode(model)
    symmetric.world_size = world
    symmetric.max_size = 64 * 2560 * 2 * world - 1
    assert not replicated_decoder.supports_replicated_decode(model)


def test_replicated_decode_keeps_unsupported_world_fallback(decode_model):
    model, _ = decode_model(2)
    assert not replicated_decoder.supports_replicated_decode(model)


@pytest.mark.parametrize("world", [4, 8])
@pytest.mark.parametrize("mtp", [2, 3, 4])
def test_fixed_attention_profile_stays_tp8(decode_model, monkeypatch, world, mtp):
    model, _ = decode_model(world, mtp=mtp)
    monkeypatch.setattr(model_module, "TritonAttBackend", lambda **kwargs: object())
    model._init_att_backend()
    assert model.fixed_decode_config == (world == 8 and mtp == 3)
