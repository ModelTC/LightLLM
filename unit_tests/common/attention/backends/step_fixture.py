from typing import Any
from dataclasses import dataclass, fields, replace
from types import SimpleNamespace
from typing import Optional
from weakref import WeakKeyDictionary

import torch

from lightllm.common.basemodel.infer_struct import GraphMode, InferStateInfo

_STEP_CONFIGS = WeakKeyDictionary()


def make_infer_state(
    *, req_to_token_indexs=None, pad_req_idx=None, cuda_graph_max_kv_seq_len=None, max_q_seq_len=1, **fields
) -> InferStateInfo:
    state = InferStateInfo()
    state.req_manager = SimpleNamespace(req_to_token_indexs=req_to_token_indexs, HOLD_REQUEST_ID=pad_req_idx)
    state.graph_mode = GraphMode.CAPTURE if cuda_graph_max_kv_seq_len is not None else GraphMode.EAGER
    state.max_q_seq_len = max_q_seq_len
    options = {}
    if cuda_graph_max_kv_seq_len is not None:
        options["cuda_graph_max_kv_seq_len"] = cuda_graph_max_kv_seq_len
    for name in ("decode_rows_per_request", "dynamic_decode_rows", "req_to_mtp_state_index"):
        if name in fields:
            options[name] = fields.pop(name)
    _STEP_CONFIGS[state] = options
    for name, value in fields.items():
        assert hasattr(state, name), name
        setattr(state, name, value)
    return state


@dataclass(frozen=True)
class AttentionStepConfig:
    """用于在没有模型的情况下构建步骤的测试的注意力输入值。"""

    # 解码步骤中每个请求的查询行；在投机解码时大于 1。
    decode_rows_per_request: int = 1
    # decode 采用动态验证时，每个请求打包的投机行数可变。
    dynamic_decode_rows: bool = False
    # 线性注意力：req_to_mtp_state_index；无循环状态时为 None。
    req_to_mtp_state_index: Optional[torch.Tensor] = None
    # 模型 CUDA Graph的 KV 长度容量；捕获和重放步骤所需。
    cuda_graph_max_kv_seq_len: Optional[int] = None


def configure_attention_step(
    infer_state: InferStateInfo, step_config: AttentionStepConfig = AttentionStepConfig()
) -> InferStateInfo:
    """记录独立后端测试所用的运行时配置，不写入推理状态。"""
    _STEP_CONFIGS[infer_state] = {field.name: getattr(step_config, field.name) for field in fields(step_config)}
    return infer_state


def create_att_state(backend: Any, infer_state: Any):
    """Configure a standalone test backend without adding production step fields to InferStateInfo."""
    from lightllm.common.attention.base_backend import PrefillAttBackend, DecodeAttBackend

    if isinstance(backend, (PrefillAttBackend, DecodeAttBackend)):
        options = dict(_STEP_CONFIGS.get(infer_state, {})) if isinstance(infer_state, InferStateInfo) else {}
        index = options.pop("req_to_mtp_state_index", None)
        if options:
            runtime = backend.runtime_config
            backend.runtime_config = replace(
                runtime,
                decode_rows_per_request=options.get("decode_rows_per_request", runtime.decode_rows_per_request),
                dynamic_decode_rows=options.get("dynamic_decode_rows", runtime.dynamic_decode_rows),
                cuda_graph_max_kv_seq_len=options.get("cuda_graph_max_kv_seq_len", runtime.cuda_graph_max_kv_seq_len),
            )
        if index is not None and infer_state.att_state_manager is None:
            infer_state.att_state_manager = SimpleNamespace(
                get_state=lambda model_role, state_type_filter, logical_layer_id: SimpleNamespace(mtp_state_index=index)
            )
        if backend.runtime_config.decode_rows_per_request > 1 and not backend.logical_layer_ids:
            backend.runtime_config = replace(backend.runtime_config, logical_layer_ids=(0,))
    att_state = backend.create_att_state(infer_state=infer_state)
    if isinstance(backend, (PrefillAttBackend, DecodeAttBackend)) and isinstance(infer_state, InferStateInfo):
        infer_state.att_states[backend] = att_state
    return att_state


def make_backend_model() -> SimpleNamespace:
    """提供真实 selector 初始化所读取的模型属性。"""
    from lightllm.server.core.objs.start_args_type import StartArgs

    return SimpleNamespace(
        args=StartArgs(running_max_req_size=8, page_size=1, flashinfer_workspace_size=128),
        att_state_role="main",
        is_mtp_draft_model=False,
        max_seq_length=512,
        graph_max_batch_size=8,
        graph_max_len_in_batch=512,
        tp_world_size_=1,
    )
