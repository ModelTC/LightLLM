from dataclasses import dataclass
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

rpyc_config = {
    "allow_pickle": True,
    "allow_all_attrs": True,
    "allow_getattr": True,
    "allow_setattr": True,
    # Bound *all* synchronous RPyC calls served on this config so a hung peer cannot
    # leave callers blocked forever (2026-05-09 incident). Applies in both directions:
    #   - client → server: enqueue / one-shot RPCs (proxy_manager → remote visual server)
    #   - server → client: worker netref calls back into manager (VisualInferResult.mark_*)
    # 30s is comfortably above the per-batch latency we see in practice; init_model and
    # run_task callers further pass explicit ans.wait timeouts when they need different
    # budgets (init is slow, run_task should be fast).
    "sync_request_timeout": 30,
}


@dataclass
class VIT_Obj:
    node_id: int
    host_ip: str
    port: int

    def to_log_str(self):
        return f"VIT host_ip_port: {self.host_ip}:{self.port}, node_id: {self.node_id}"
