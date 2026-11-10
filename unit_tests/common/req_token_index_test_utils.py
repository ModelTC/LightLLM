"""Attach the production KV allocation and token-index APIs to test requests."""

from types import MethodType, SimpleNamespace

import torch

from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.infer_req_ext1 import (
    PinMemPageCacheReqTokenIndex,
    create_req_token_index,
)


def attach_req_token_index(req, req_manager, token_indexes=None, *, mode="pin_mem_cache"):
    req.req_manager = req_manager
    req.args = SimpleNamespace(page_size=req_manager.page_size, req_token_index_mode=mode)
    # Tests exercise several service configurations in one process.
    offsets = PinMemPageCacheReqTokenIndex._page_offsets
    if offsets is not None and offsets.numel() != req_manager.page_size:
        PinMemPageCacheReqTokenIndex._page_offsets = None
    req.token_index = create_req_token_index(req, torch.cuda.current_stream())
    req.kv_cache_alloc_need = MethodType(InferReq.kv_cache_alloc_need, req)
    req.alloc_kv_mem = MethodType(InferReq.alloc_kv_mem, req)
    if token_indexes is not None:
        assert token_indexes.numel() == req.hold_kv_len
        req.token_index.append_token_indexes(token_indexes)
    return req
