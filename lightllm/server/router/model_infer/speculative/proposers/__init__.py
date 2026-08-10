from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.speculative.proposers.base import BaseSpecProposer


def build_spec_proposer(engine) -> "BaseSpecProposer":
    spec_mode = engine.spec_mode
    if spec_mode == "dspark":
        from lightllm.server.router.model_infer.speculative.proposers.dspark import DSparkProposer

        return DSparkProposer(engine=engine)
    if spec_mode == "dflash":
        from lightllm.server.router.model_infer.speculative.proposers.dflash import DFlashProposer

        return DFlashProposer(engine=engine)
    if spec_mode == "eagle3":
        from lightllm.server.router.model_infer.speculative.proposers.eagle3 import Eagle3Proposer

        return Eagle3Proposer(engine=engine)
    if spec_mode in ("eagle_with_att", "eagle_no_att", "qwen3next_eagle"):
        from lightllm.server.router.model_infer.speculative.proposers.eagle_mtp import EagleMTPProposer

        return EagleMTPProposer(engine=engine)

    from lightllm.server.router.model_infer.speculative.proposers.vanilla_mtp import VanillaMTPProposer

    return VanillaMTPProposer(engine=engine)


__all__ = [
    "build_spec_proposer",
]
