"""GDN 运行态及其 SSM 执行策略。"""

from ..conv_ssm_state import _ConvSSMState
from .gdn_pages_pool import GDNPagesPool


class GDNState(_ConvSSMState):
    """GDN 的公共资源外壳；具体 replay 策略由下列子类固定。"""

    algorithm = "gdn"
    pages_pool_class = GDNPagesPool


class GDNNativeState(GDNState):
    """GDN 原生 MTP 槽位策略。"""

    state_mode = "native"


class GDNReplayState(GDNState):
    """GDN replay history 策略。"""

    state_mode = "replay"
