__all__ = [
    "SpecRuntime",
    "SpecDecodeForwardState",
    "SpecDecodePostState",
    "SpecDecodeRunner",
    "SpecVerifier",
    "SpecVerifyResult",
    "build_spec_runtime",
]


def __getattr__(name):
    if name in ("SpecRuntime", "build_spec_runtime"):
        from lightllm.server.router.model_infer.speculative.runtime import SpecRuntime, build_spec_runtime

        values = {
            "SpecRuntime": SpecRuntime,
            "build_spec_runtime": build_spec_runtime,
        }
        return values[name]
    if name in ("SpecDecodeForwardState", "SpecDecodePostState", "SpecDecodeRunner"):
        from lightllm.server.router.model_infer.speculative.runner import (
            SpecDecodeForwardState,
            SpecDecodePostState,
            SpecDecodeRunner,
        )

        values = {
            "SpecDecodeForwardState": SpecDecodeForwardState,
            "SpecDecodePostState": SpecDecodePostState,
            "SpecDecodeRunner": SpecDecodeRunner,
        }
        return values[name]
    if name in ("SpecVerifier", "SpecVerifyResult"):
        from lightllm.server.router.model_infer.speculative.verifier import SpecVerifier, SpecVerifyResult

        values = {
            "SpecVerifier": SpecVerifier,
            "SpecVerifyResult": SpecVerifyResult,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
