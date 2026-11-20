"""PD 分离的 Prefill 先推理模式里，推理进程用的占页和传输。"""

import pickle
from typing import List, Tuple

from lightllm.server.pd_io_struct import PDDecodeNodeInfo
from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context


def submit_prefill_first_tasks_if_ready(
    backend, req_obj: InferReq, next_token_id: int, next_token_logprob: float
) -> None:
    """共享内存里已有 decode KV 就绪信息时，按 `ready_kv_len` 组传输任务。

    首 token 写在最后一块上。混合注意力模型的最后一块是 `att_state`。
    """
    trans_params = req_obj.sampling_param.shm_param.pd_kv_trans_params
    decode_node_info: PDDecodeNodeInfo = pickle.loads(trans_params.get())
    input_len = req_obj.shm_req.input_len
    ready_kv_len = decode_node_info.ready_kv_len
    assert input_len > ready_kv_len, (
        f"prefill-first decode kv ready has no token to send, request_id={req_obj.req_id} "
        f"ready_kv_len={ready_kv_len} input_len={input_len}"
    )

    page_size = backend.args.pd_kv_page_size
    req_obj.sampling_param.pd_decode_node = decode_node_info
    tasks = []
    for span_start, span_end in iter_kv_spans(start=ready_kv_len, end=input_len, page_size=page_size):
        tasks.append(
            backend._create_pd_trans_task(
                req_obj=req_obj,
                kv_start_index=span_start,
                kv_end_index=span_end,
            )
        )
    if g_infer_context.is_hybrid_att_model:
        tasks.append(
            backend._create_pd_trans_task(
                req_obj=req_obj,
                kv_start_index=input_len,
                kv_end_index=input_len,
                page_kind="att_state",
            )
        )

    tasks[-1].first_gen_token_id = next_token_id
    tasks[-1].first_gen_token_logprob = next_token_logprob

    if backend.is_master_in_dp:
        for task in tasks:
            backend.info_queue.put(task)
    return


def iter_kv_spans(start: int, end: int, page_size: int) -> List[Tuple[int, int]]:
    """把 `[start, end)` 切成和 `pd_kv_page_size` 对齐的半开区间。

    调用方保证 `end > start`。满匹配时 `end - start` 仍至少是 1，所以这里不会得到空列表。
    """
    if end <= start:
        raise ValueError(f"kv span requires end > start, got [{start}, {end})")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")

    spans: List[Tuple[int, int]] = []
    cursor = start
    while cursor < end:
        step = min(page_size, end - cursor)
        spans.append((cursor, cursor + step))
        cursor += step
    return spans
