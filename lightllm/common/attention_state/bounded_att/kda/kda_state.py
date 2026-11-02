from ..conv_ssm_state import ConvSSMState
from .kda_pages_pool import KDAPagesPool


class KDAState(ConvSSMState):
    """KDA convolution history, recurrent state and accepted speculative cursor."""

    algorithm = "kda"
    pool_type = KDAPagesPool
