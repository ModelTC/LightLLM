import json
from types import SimpleNamespace
from contextlib import nullcontext
from uuid import uuid4

import pytest
import torch

from lightllm.models.qwen4_exp.layer_weights.transformer_layer_weight import Qwen4ExpTransformerLayerWeight
from lightllm.models.qwen4_exp.infer_struct import Qwen4ExpInferStateInfo
from lightllm.models.qwen4_exp.model import Qwen4ExpTpPartModel
from lightllm.common.quantization import Quantcfg
from lightllm.utils import envs_utils
from .helpers import checkpoint, prepost_checkpoint


@pytest.fixture
def runtime(config, tmp_path, monkeypatch, request):
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = envs_utils.get_env_start_args()
    args.model_dir = str(tmp_path)
    args.data_type = "bf16"
    for name, value in getattr(request, "param", {}).items():
        setattr(args, name, value)
    args.cpu_cache_token_page_size = args.linear_att_hash_page_size * args.linear_att_page_block_num
    args.linear_att_ssm_data_type = "float32"
    args.llm_prefill_att_backend = ["triton", "triton"]
    args.llm_decode_att_backend = ["triton", "triton"]
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(vars(args)))
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_NODE", "0")
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", "qwen-runtime-test-" + uuid4().hex)
    envs_utils.get_env_start_args.cache_clear()
    envs_utils.get_unique_server_name.cache_clear()
    weight = Qwen4ExpTransformerLayerWeight(1, torch.bfloat16, config, Quantcfg(config))
    model = Qwen4ExpTpPartModel.__new__(Qwen4ExpTpPartModel)
    model.__dict__.update(
        config=config,
        args=args,
        tp_world_size_=1,
        tp_rank_=0,
        torch_memory_saver=SimpleNamespace(region=lambda **kwargs: nullcontext()),
        data_type=torch.bfloat16,
        max_req_num=4,
        max_seq_length=64,
        batch_max_tokens=8,
        max_total_token_num=32,
        num_kv_heads=2,
        mem_fraction=0.9,
        trans_layers_weight=[weight],
    )
    from lightllm.common.req_manager import ReqManager
    from .resource_helpers import bind_states

    model.layers_infer = [SimpleNamespace(layer_num_=i) for i in range(config["n_layer"])]
    req = ReqManager(4, 64)
    model.bind_req_manager(req)
    model.prepare_attention_states()
    mem = bind_states(model)
    assert not weight.ngram_hasher.verify_load()
    torch.manual_seed(1307)
    source = checkpoint(config, 1)
    weight.load_hf_weights(dict(source))
    weight.verify_load()
    angle = torch.randn(64, 2, device="cuda")

    model._cos_cached = angle.cos().bfloat16()
    model._sin_cached = angle.sin().bfloat16()
    model.mtp_manager = SimpleNamespace(get_decode_draft_step=lambda is_draft: 0)
    model._init_att_backend()
    model._init_att_backend1()
    yield SimpleNamespace(mem=mem, req=req, weight=weight, source=source, model=model)
    envs_utils.get_unique_server_name.cache_clear()


def state_for(runtime, requests, lengths, tokens, ready=None, mtp_indices=None):
    state = Qwen4ExpInferStateInfo()
    state.req_manager = runtime.req
    state.att_state_manager = runtime.model.att_state_manager
    state.att_state_role = runtime.model.att_state_role
    state.is_prefill = ready is not None
    state.batch_size = len(requests)
    state.b_req_idx = torch.tensor(requests, device="cuda", dtype=torch.int32)
    state.b_seq_len = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    if mtp_indices is not None:
        state.b_mtp_index = torch.tensor(mtp_indices, device="cuda", dtype=torch.int32)
    state.input_ids = torch.tensor(tokens, device="cuda", dtype=torch.int64)
    state.mem_index = runtime.mem.alloc(len(tokens)).cuda(non_blocking=True)
    if ready is not None:
        state.b_ready_cache_len = torch.tensor(ready, device="cuda", dtype=torch.int32)
        query_lengths = [length - prefix for length, prefix in zip(lengths, ready)]
    else:
        state.b_position_delta = torch.zeros(len(requests), device="cuda", dtype=torch.int64)
        query_lengths = [1] * len(requests)
    offset = 0
    for request, length, query in zip(requests, lengths, query_lengths):
        runtime.req.req_to_token_indexs[request, length - query : length] = state.mem_index[offset : offset + query]
        offset += query
    state.multimodal_params = []
    state.init_some_extra_state(runtime.model)
    if state.is_prefill:
        state.prefill_att_state = runtime.model.prefill_att_backend.create_att_prefill_state(state)
        if runtime.model.prefill_att_backend1 is not None:
            state.prefill_att_state1 = runtime.model.prefill_att_backend1.create_att_prefill_state(state)
    else:
        state.decode_att_state = runtime.model.decode_att_backend.create_att_decode_state(state)
        if runtime.model.decode_att_backend1 is not None:
            state.decode_att_state1 = runtime.model.decode_att_backend1.create_att_decode_state(state)
    state.init_att_state()
    return state


def assemble_model(runtime, config):
    model, source = runtime.model, prepost_checkpoint(config)
    weights = []
    for layer in range(config["n_layer"]):
        if layer == 1:
            weight = runtime.weight
            source.update(runtime.source)
        else:
            local = checkpoint(config, layer)
            source.update(local)
            weight = model.transformer_weight_class(layer, torch.bfloat16, config, Quantcfg(config))
            weight.load_hf_weights(dict(local))
            weight.verify_load()
        weights.append(weight)
    model.trans_layers_weight = weights
    model.pre_post_weight = model.pre_and_post_weight_class(torch.bfloat16, config)
    model.pre_post_weight.load_hf_weights(dict(source))
    model.pre_post_weight.verify_load()
    model._init_infer_layer()
    model._init_some_value()
    model.prefill_graph = None
    return model, source
