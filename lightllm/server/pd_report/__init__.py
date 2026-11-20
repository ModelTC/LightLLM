"""Decode 节点上把推理上报转给对应的 PD Master。"""

from lightllm.server.pd_report.manager import start_pd_report_process

__all__ = ["start_pd_report_process"]
