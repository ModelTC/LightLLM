from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
    from lightllm.server.router.model_infer.mtp_speculative.dp_proposers.base import BaseDpProposer


def build_dp_spec_proposer(*, spec_mode: str, backend: "ModeBackend", enable_dynmaic_mtp: bool) -> "BaseDpProposer":
    if spec_mode == "vanilla_with_att":
        from lightllm.server.router.model_infer.mtp_speculative.dp_proposers.vanilla_with_att import (
            DpVanillaWithAttProposer,
        )

        return DpVanillaWithAttProposer(backend=backend, enable_dynmaic_mtp=enable_dynmaic_mtp)
    if spec_mode == "vanilla_no_att":
        from lightllm.server.router.model_infer.mtp_speculative.dp_proposers.vanilla_no_att import (
            DpVanillaNoAttProposer,
        )

        return DpVanillaNoAttProposer(backend=backend, enable_dynmaic_mtp=enable_dynmaic_mtp)
    if spec_mode == "eagle_with_att":
        from lightllm.server.router.model_infer.mtp_speculative.dp_proposers.eagle_with_att import (
            DpEagleWithAttProposer,
        )

        return DpEagleWithAttProposer(backend=backend, enable_dynmaic_mtp=enable_dynmaic_mtp)
    if spec_mode == "eagle_no_att":
        from lightllm.server.router.model_infer.mtp_speculative.dp_proposers.eagle_no_att import DpEagleNoAttProposer

        return DpEagleNoAttProposer(backend=backend, enable_dynmaic_mtp=enable_dynmaic_mtp)
    if spec_mode == "eagle3":
        from lightllm.server.router.model_infer.mtp_speculative.dp_proposers.eagle3 import DpEagle3Proposer

        return DpEagle3Proposer(backend=backend, enable_dynmaic_mtp=enable_dynmaic_mtp)

    raise ValueError(f"unsupported DP speculative mode: {spec_mode}")


__all__ = ["build_dp_spec_proposer"]
