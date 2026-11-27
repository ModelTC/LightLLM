"""最后一段 prefill 被调度时，经本机 pd_report 把 prompt 交给 PD Master。"""

from lightllm.server.pd_io_struct import PdPfPrefillKvReady
from lightllm.server.pd_report.ipc import ReportIpc
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

_prefill_kv_ready_push = None


def push_prefill_first_kv_ready(req: InferReq) -> None:
    """把 prompt 推给本机 pd_report。由它按 pd_master_node_id 写到对应 Master 的 /pd_report。"""
    ready = PdPfPrefillKvReady(
        group_request_id=int(req.shm_req.sample_params.group_request_id),
        prompt_ids=[int(token_id) for token_id in req.shm_req.get_prompt_ids()],
        pd_master_node_id=int(req.sampling_param.pd_master_node_id),
    )
    _sender().send(message=ready)
    logger.info(
        f"prefill-first stage=report-prefill-kv-ready, group_request_id={ready.group_request_id}, "
        f"prompt_len={len(ready.prompt_ids)}"
    )


def _sender() -> ReportIpc.Push:
    global _prefill_kv_ready_push
    if _prefill_kv_ready_push is None:
        _prefill_kv_ready_push = ReportIpc.Push()
    return _prefill_kv_ready_push
