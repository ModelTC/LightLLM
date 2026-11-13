from lightllm.platform import get_device_runtime

from .api import AttentionFamily, AttentionPhase, UnsupportedAttention
from .registry import AUDIO_VARLEN_PRIORITY, get_varlen_attention_op_class
from .selection import BackendSelectionError


class AutoVarlenAttentionRunner:
    """One encoder owns this runner and shares stateless operators across its layers.

    Unsupported inputs can try the next provider for this call. Kernel and programming
    errors propagate: retrying on a potentially failed device is not a safe fallback.
    Environment rejections are cached by provider/device; input rejections are not.
    """

    def __init__(self, providers=AUDIO_VARLEN_PRIORITY):
        self.providers = providers
        self.ops = {}
        self.unavailable = {}

    def forward(self, inputs, *, out):
        failures = {}
        caps = get_device_runtime(inputs.query.device).caps
        for name in self.providers:
            key = (name, inputs.query.device)
            if key in self.unavailable:
                failures[name] = self.unavailable[key]
                continue
            if key not in self.ops:
                try:
                    op_cls = get_varlen_attention_op_class(name)
                    op_cls.check_environment(caps)
                    self.ops[key] = op_cls()
                except (UnsupportedAttention, ImportError) as exc:
                    self.unavailable[key] = failures[name] = str(exc)
                    continue
            op = self.ops[key]
            try:
                op.check_support(inputs, caps)
            except UnsupportedAttention as exc:
                failures[name] = str(exc)
                continue
            return op.forward(inputs, out=out)
        raise BackendSelectionError(AttentionFamily.DENSE, AttentionPhase.PREFILL, failures)
