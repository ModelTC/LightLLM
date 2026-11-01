from functools import partial
import torch
from ..conv_ssm_types import ConvSSMStateConfig
from ..conv_ssm_state import ConvSSMState
from ..formats import conv_ssm_spec
from .gdn_pages_pool import GDNPagesPool


class GDNState(ConvSSMState):
    def __init__(
        self, config: ConvSSMStateConfig, request_capacity: int, mtp_step: int = 0, device: str | torch.device = "cuda"
    ):
        from lightllm.common.attention_compute.bounded import ConvSSMRuntime

        assert isinstance(config, ConvSSMStateConfig), "config must be ConvSSMStateConfig"
        spec = conv_ssm_spec(config=config, request_capacity=request_capacity, mtp_step=mtp_step, algorithm="gdn")
        super().__init__(
            config=config,
            spec=spec,
            runtime=ConvSSMRuntime(config),
            request_capacity=request_capacity,
            mtp_step=mtp_step,
            device=device,
            page_pool_factory=partial(GDNPagesPool, config=config, snapshot_fields=spec.snapshot_fields),
        )
