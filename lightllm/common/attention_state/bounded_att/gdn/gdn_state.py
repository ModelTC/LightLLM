from ..conv_ssm_state import ConvSSMState
from .gdn_pages_pool import GDNPagesPool


class GDNState(ConvSSMState):
    """Gated DeltaNet convolution history, recurrent state and accepted speculative cursor."""

    algorithm = "gdn"
    pool_type = GDNPagesPool
