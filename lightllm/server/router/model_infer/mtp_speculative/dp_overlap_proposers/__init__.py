from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
    from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.base import BaseDpOverlapProposer


def build_dp_overlap_spec_proposer(
    *,
    spec_mode: str,
    backend: "ModeBackend",
    enable_dynmaic_mtp: bool,
) -> "BaseDpOverlapProposer":
    if spec_mode == "vanilla_with_att":
        from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.vanilla_with_att import (
            DpOverlapVanillaWithAttProposer,
        )

        return DpOverlapVanillaWithAttProposer(backend=backend, enable_dynmaic_mtp=enable_dynmaic_mtp)
    if spec_mode == "vanilla_no_att":
        from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.vanilla_no_att import (
            DpOverlapVanillaNoAttProposer,
        )

        return DpOverlapVanillaNoAttProposer(backend=backend, enable_dynmaic_mtp=enable_dynmaic_mtp)
    if spec_mode == "eagle_with_att":
        from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.eagle_with_att import (
            DpOverlapEagleWithAttProposer,
        )

        return DpOverlapEagleWithAttProposer(backend=backend, enable_dynmaic_mtp=enable_dynmaic_mtp)
    if spec_mode == "eagle_no_att":
        from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.eagle_no_att import (
            DpOverlapEagleNoAttProposer,
        )

        return DpOverlapEagleNoAttProposer(backend=backend, enable_dynmaic_mtp=enable_dynmaic_mtp)
    if spec_mode == "eagle3":
        from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.eagle3 import (
            DpOverlapEagle3Proposer,
        )

        return DpOverlapEagle3Proposer(backend=backend, enable_dynmaic_mtp=enable_dynmaic_mtp)

    raise ValueError(f"unsupported DP overlap speculative mode: {spec_mode}")


__all__ = ["build_dp_overlap_spec_proposer"]
