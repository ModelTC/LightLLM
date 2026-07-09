from lightllm.server.router.model_infer.speculative.proposers.base import BaseSpecProposer, SpecProposal


def build_spec_proposer(runtime) -> BaseSpecProposer:
    spec_config = runtime.spec_config
    if spec_config.is_dspark:
        from lightllm.server.router.model_infer.speculative.proposers.dspark import DSparkProposer

        return DSparkProposer(runtime)
    if spec_config.is_dflash:
        from lightllm.server.router.model_infer.speculative.proposers.dflash import DFlashProposer

        return DFlashProposer(runtime)
    if spec_config.is_eagle3:
        from lightllm.server.router.model_infer.speculative.proposers.eagle3 import Eagle3Proposer

        return Eagle3Proposer(runtime)
    if spec_config.uses_recurrent_draft_model:
        from lightllm.server.router.model_infer.speculative.proposers.eagle_mtp import EagleMTPProposer

        return EagleMTPProposer(runtime)

    from lightllm.server.router.model_infer.speculative.proposers.vanilla_mtp import VanillaMTPProposer

    return VanillaMTPProposer(runtime)


def __getattr__(name):
    if name == "DFlashProposer":
        from lightllm.server.router.model_infer.speculative.proposers.dflash import DFlashProposer

        return DFlashProposer
    if name == "DSparkProposer":
        from lightllm.server.router.model_infer.speculative.proposers.dspark import DSparkProposer

        return DSparkProposer
    if name == "EagleMTPProposer":
        from lightllm.server.router.model_infer.speculative.proposers.eagle_mtp import EagleMTPProposer

        return EagleMTPProposer
    if name == "Eagle3Proposer":
        from lightllm.server.router.model_infer.speculative.proposers.eagle3 import Eagle3Proposer

        return Eagle3Proposer
    if name == "VanillaMTPProposer":
        from lightllm.server.router.model_infer.speculative.proposers.vanilla_mtp import VanillaMTPProposer

        return VanillaMTPProposer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseSpecProposer",
    "DFlashProposer",
    "DSparkProposer",
    "EagleMTPProposer",
    "Eagle3Proposer",
    "SpecProposal",
    "VanillaMTPProposer",
    "build_spec_proposer",
]
