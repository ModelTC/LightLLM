from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
    from lightllm.server.router.model_infer.mtp_speculative.proposers.base import BaseSpecProposer


def build_spec_proposer(*, spec_mode: str, backend: "ModeBackend", enable_dynmaic_mtp: bool) -> "BaseSpecProposer":
    if spec_mode == "dspark":
        from lightllm.server.router.model_infer.mtp_speculative.proposers.dspark import DSparkProposer

        return DSparkProposer(backend=backend, enable_dynmaic_mtp=enable_dynmaic_mtp)
    if spec_mode == "dflash":
        from lightllm.server.router.model_infer.mtp_speculative.proposers.dflash import DFlashProposer

        return DFlashProposer(backend=backend, enable_dynmaic_mtp=enable_dynmaic_mtp)
    if spec_mode == "eagle3":
        from lightllm.server.router.model_infer.mtp_speculative.proposers.eagle3 import Eagle3Proposer

        return Eagle3Proposer(backend=backend, enable_dynmaic_mtp=enable_dynmaic_mtp)
    if spec_mode == "eagle_with_att":
        from lightllm.server.router.model_infer.mtp_speculative.proposers.eagle_with_att import EagleWithAttProposer

        return EagleWithAttProposer(backend=backend, enable_dynmaic_mtp=enable_dynmaic_mtp)
    if spec_mode == "eagle_no_att":
        from lightllm.server.router.model_infer.mtp_speculative.proposers.eagle_no_att import EagleNoAttProposer

        return EagleNoAttProposer(backend=backend, enable_dynmaic_mtp=enable_dynmaic_mtp)
    if spec_mode == "vanilla_with_att":
        from lightllm.server.router.model_infer.mtp_speculative.proposers.vanilla_with_att import (
            VanillaWithAttProposer,
        )

        return VanillaWithAttProposer(backend=backend, enable_dynmaic_mtp=enable_dynmaic_mtp)
    if spec_mode == "vanilla_no_att":
        from lightllm.server.router.model_infer.mtp_speculative.proposers.vanilla_no_att import VanillaNoAttProposer

        return VanillaNoAttProposer(backend=backend, enable_dynmaic_mtp=enable_dynmaic_mtp)

    raise ValueError(f"unsupported speculative mode: {spec_mode}")


__all__ = [
    "build_spec_proposer",
]
