from lightllm.platform import get_device_runtime
from lightllm.utils.log_utils import init_logger

from .api import AttentionFamily, AttentionPhase, UnsupportedAttention
from .registry import AUDIO_VARLEN_PRIORITY, get_varlen_attention_op_class
from .selection import BackendSelectionError

logger = init_logger(__name__)


class AutoVarlenAttentionRunner:
    """音频编码器的自动路径；失败的实现只在当前调用方禁用，并在本次调用中重算输出。"""

    def __init__(self, providers=AUDIO_VARLEN_PRIORITY):
        self.providers = providers
        self.ops = {}
        self.failures = {}

    def forward(self, inputs, *, out):
        for name in self.providers:
            if name in self.failures:
                continue
            try:
                if name not in self.ops:
                    op_cls = get_varlen_attention_op_class(name)
                    caps = get_device_runtime(inputs.query.device).caps
                    op_cls.check_environment(caps)
                    op_cls.check_support(inputs, caps)
                    self.ops[name] = op_cls()
                return self.ops[name].forward(inputs, out=out)
            except UnsupportedAttention as exc:
                self.failures[name] = str(exc)
            except Exception as exc:
                self.failures[name] = str(exc)
                logger.warning("Disabling %s varlen attention for this encoder: %s", name, exc)
        raise BackendSelectionError(AttentionFamily.DENSE, AttentionPhase.PREFILL, self.failures)
