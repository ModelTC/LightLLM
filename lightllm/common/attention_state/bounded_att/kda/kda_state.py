"""KDA 运行态及其三种 SSM 执行策略。"""

from ..conv_ssm_state import _ConvSSMState
from .kda_pages_pool import KDAPagesPool


class KDAState(_ConvSSMState):
    """KDA 的公共资源外壳；具体 replay 策略由下列子类固定。"""

    algorithm = "kda"
    pages_pool_class = KDAPagesPool


class KDANativeState(KDAState):
    """KDA 原生 MTP 槽位策略。"""

    state_mode = "native"


class KDACompactState(KDAState):
    """KDA compact 接受前缀重建策略。"""

    state_mode = "compact"


class KDAReplayState(KDAState):
    """KDA replay history 策略。"""

    state_mode = "replay"
