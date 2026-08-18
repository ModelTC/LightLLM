from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.speculative.proposers.base import BaseSpecProposer


def build_spec_proposer(*, spec_mode: str, backend, enable_dynmaic_mtp: bool) -> "BaseSpecProposer":
    if spec_mode == "dspark":
        from lightllm.server.router.model_infer.speculative.proposers.dspark import DSparkProposer

        return DSparkProposer(backend=backend, enable_dynmaic_mtp=enable_dynmaic_mtp)
    if spec_mode == "dflash":
        from lightllm.server.router.model_infer.speculative.proposers.dflash import DFlashProposer

        return DFlashProposer(backend=backend, enable_dynmaic_mtp=enable_dynmaic_mtp)
    if spec_mode == "eagle3":
        from lightllm.server.router.model_infer.speculative.proposers.eagle3 import Eagle3Proposer

        return Eagle3Proposer(backend=backend, enable_dynmaic_mtp=enable_dynmaic_mtp)
    if spec_mode in ("eagle_with_att", "eagle_no_att"):
        from lightllm.server.router.model_infer.speculative.proposers.eagle_mtp import EagleMTPProposer

        return EagleMTPProposer(backend=backend, enable_dynmaic_mtp=enable_dynmaic_mtp)
    if spec_mode in ("vanilla_with_att", "vanilla_no_att"):
        from lightllm.server.router.model_infer.speculative.proposers.vanilla_mtp import VanillaMTPProposer

        return VanillaMTPProposer(backend=backend, enable_dynmaic_mtp=enable_dynmaic_mtp)

    raise ValueError(f"unsupported speculative mode: {spec_mode}")


__all__ = [
    "build_spec_proposer",
]
