from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from .basemodel import TpPartBaseModel


def initialize_model_group(main_model: TpPartBaseModel, draft_models: Sequence[TpPartBaseModel] = ()) -> None:
    from lightllm.utils.envs_utils import set_model_init_status
    from lightllm.utils.torch_memory_saver_utils import MemoryTag
    from lightllm.common.attention_state import AttentionStateManager
    from lightllm.common.req_manager import ReqManager

    models = (main_model, *draft_models)
    assert all(getattr(item, "att_state_manager", None) is None for item in models), "resources already initialized"
    assert all(item.max_req_num == main_model.max_req_num for item in models), "models must share the request domain"
    # main/draft 共用请求映射；按所有模型的最大需求分配，再逐模型绑定。
    max_seq_len = max(item.get_req_manager_max_seq_len() for item in models)
    with main_model.torch_memory_saver.region(tag=MemoryTag.KV_CACHE):
        req_manager = ReqManager(max_request_num=main_model.max_req_num, max_sequence_length=max_seq_len)
    for item in models:
        item.bind_req_manager(req_manager=req_manager)
    for item in models:
        item.prepare_attention_states()
    manager: AttentionStateManager = main_model.att_state_manager
    assert all(item.att_state_manager is manager for item in models), "models must share one resource manager"
    with main_model.torch_memory_saver.region(tag=MemoryTag.KV_CACHE):
        manager.finish_state_registration()
        manager.load_full_att_calibration(args=main_model.args)
        manager.initialize_storage(
            main_model.args, token_capacity=main_model.max_total_token_num, mem_fraction=main_model.mem_fraction
        )
    for item in models:
        item.init_runtime()
    torch.cuda.empty_cache()
    set_model_init_status(True)
