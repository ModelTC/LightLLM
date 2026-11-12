from lightllm.platform import get_device_runtime
from lightllm.utils.log_utils import init_logger

from .api import AttentionFamily, AttentionPhase
from .selection import BackendSelectionError, get_varlen_backend_class

logger = init_logger(__name__)


class AutoVarlenAttention:
    """音频编码器的自动路径；失败的实现只在当前调用方禁用，并在本次调用中重算输出。"""

    def __init__(self, providers=("fa3", "triton")):
        self.providers = providers
        self.backends = {}
        self.failures = {}

    def forward(self, inputs, *, out):
        for name in self.providers:
            if name in self.failures:
                continue
            try:
                if name not in self.backends:
                    backend = get_varlen_backend_class(name)
                    caps = get_device_runtime(inputs.query.device).caps
                    environment = backend.check_environment(caps)
                    if not environment.supported:
                        self.failures[name] = environment.reason
                        continue
                    support = backend.check_support(inputs, caps)
                    if not support.supported:
                        self.failures[name] = support.reason
                        continue
                    self.backends[name] = backend()
                return self.backends[name].forward(inputs, out=out)
            except Exception as exc:
                self.failures[name] = str(exc)
                logger.warning("Disabling %s varlen attention for this encoder: %s", name, exc)
        raise BackendSelectionError(AttentionFamily.DENSE, AttentionPhase.PREFILL, self.failures)
