from abc import ABC

from lightllm.server.router.model_infer.mtp_speculative.proposers.base import BaseSpecProposer


class BaseDpProposer(BaseSpecProposer, ABC):
    """普通 DP prefill/decode proposer 的基础接口。"""
