"""Autonomous VLA and media generation servers.

Shared activation ownership lives in ``lightllm.server.activation_transfer``.
There is no common generation orchestrator; every modality server owns its
own scheduling semantics.
"""

from lightllm.server.activation_transfer import ConsumerExecutionError

__all__ = ["ConsumerExecutionError"]
