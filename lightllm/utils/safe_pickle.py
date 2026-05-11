"""
safe_pickle.py — Restricted unpickling for LightLLM PD WebSocket endpoints.

CVE-2026-26220: The PD disaggregation WebSocket endpoints and the config-server
HTTP response handler called pickle.loads() on untrusted network data, enabling
unauthenticated remote code execution.  This module replaces those bare
pickle.loads() calls with a RestrictedUnpickler that whitelists only the
internal LightLLM dataclass modules that legitimately flow through those
channels.

Usage:
    from lightllm.utils.safe_pickle import safe_loads

    obj = safe_loads(data)   # raises UnpicklingError for non-whitelisted types
"""

import io
import pickle

# ---------------------------------------------------------------------------
# Allowlist: (module, name) pairs permitted to be deserialized.
# Keep this list minimal — add entries only when a new type is deliberately
# added to a PD WebSocket protocol message.
# ---------------------------------------------------------------------------
_ALLOWED_MODULES: dict[str, set[str]] = {
    # Built-in safe types used as containers / primitives
    "builtins": {"dict", "list", "tuple", "set", "int", "float", "str", "bool", "bytes", "NoneType"},
    # LightLLM PD protocol structures
    "lightllm.server.pd_io_struct": {
        "ObjType",
        "NodeRole",
        "PD_Master_Obj",
        "PD_Client_Obj",
        "_PD_Client_RunStatus",
        "UpKVStatus",
        "NixlUpKVStatus",
        "DecodeNodeInfo",
        "NIXLDecodeNodeInfo",
        "KVMoveTask",
        "KVMoveTaskGroup",
        "PDTransJoinInfo",
        "PDTransLeaveInfo",
        "NIXLChunckedTransTask",
        "NIXLChunckedTransTaskRet",
        "NIXLChunckedTransTaskGroup",
    },
    # SamplingParams / StartArgs sent inside REQ messages
    "lightllm.server.core.objs.py_sampling_params": {"SamplingParams"},
    "lightllm.server.core.objs.sampling_params": {"SamplingParams"},
    "lightllm.server.core.objs.start_args_type": {"StartArgs"},
    # Multimodal params can accompany REQ messages
    "lightllm.server.multimodal_params": {"MultimodalParams"},
    # enum base class (needed for ObjType / NodeRole reconstruction)
    "enum": {"Enum"},
}


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that raises UnpicklingError for any non-whitelisted class."""

    def find_class(self, module: str, name: str):
        allowed_names = _ALLOWED_MODULES.get(module)
        if allowed_names is not None and name in allowed_names:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Refusing to deserialize {module}.{name}: not in safe_pickle allowlist. "
            f"This may indicate an attempted pickle-based RCE exploit (CVE-2026-26220)."
        )


def safe_loads(data: bytes) -> object:
    """
    Drop-in replacement for pickle.loads() that only permits whitelisted types.

    Raises pickle.UnpicklingError if the payload attempts to instantiate a
    class outside the allowlist.
    """
    return _RestrictedUnpickler(io.BytesIO(data)).load()
