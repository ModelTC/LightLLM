from ..conv_ssm_state import ConvSSMState
from .kda_pages_pool import KDAPagesPool


class KDAState(ConvSSMState):

    algorithm = "kda"
    pages_pool_class = KDAPagesPool
