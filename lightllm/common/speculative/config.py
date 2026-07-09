from dataclasses import dataclass
from typing import Any, Mapping, Optional


VANILLA_SPEC_MODES = frozenset({"vanilla_with_att", "vanilla_no_att", "qwen3next_vanilla"})
EAGLE_SPEC_MODES = frozenset({"eagle_with_att", "eagle_no_att", "eagle3", "qwen3next_eagle"})
BLOCK_SPEC_MODES = frozenset({"dspark", "dflash"})
SPEC_MODES = VANILLA_SPEC_MODES | EAGLE_SPEC_MODES | BLOCK_SPEC_MODES

ATTENTION_SPEC_MODES = frozenset({"vanilla_with_att", "eagle_with_att", "eagle3", "dspark", "dflash"})
NO_ATTENTION_SPEC_MODES = frozenset({"vanilla_no_att", "eagle_no_att", "qwen3next_vanilla", "qwen3next_eagle"})
TARGET_HIDDEN_SPEC_MODES = frozenset({"eagle3", "dspark", "dflash"})
QWEN3_DFLASH_ARCHITECTURES = frozenset({"Qwen3DFlashModel", "Qwen3DSparkModel"})
QWEN3_DSPARK_ARCHITECTURES = frozenset({"Qwen3DSparkModel"})
GEMMA4_DSPARK_ARCHITECTURES = frozenset({"Gemma4DSparkModel"})
DSPARK_FAMILY_ARCHITECTURES = QWEN3_DFLASH_ARCHITECTURES | GEMMA4_DSPARK_ARCHITECTURES
DSPARK_MARKOV_HEAD_TYPES = frozenset({"vanilla", "gated", "rnn"})


@dataclass(frozen=True)
class SpeculativeConfig:
    """Normalized view of speculative decoding mode flags."""

    mode: Optional[str]
    step: int
    dynamic_verify: bool = False

    @classmethod
    def from_args(cls, args: Any, dynamic_verify: Optional[bool] = None) -> "SpeculativeConfig":
        mode = getattr(args, "mtp_mode", None)
        if dynamic_verify is None:
            dynamic_verify = bool(getattr(args, "mtp_dynamic_verify", False))
        if mode == "dspark":
            dynamic_verify = True
        elif mode == "dflash":
            dynamic_verify = False
        return cls(
            mode=mode,
            step=int(getattr(args, "mtp_step", 0)),
            dynamic_verify=dynamic_verify,
        )

    @property
    def enabled(self) -> bool:
        return self.mode is not None

    @property
    def is_vanilla(self) -> bool:
        return self.mode in VANILLA_SPEC_MODES

    @property
    def is_eagle(self) -> bool:
        return self.mode in EAGLE_SPEC_MODES

    @property
    def is_eagle3(self) -> bool:
        return self.mode == "eagle3"

    @property
    def is_dspark(self) -> bool:
        return self.mode == "dspark"

    @property
    def is_dflash(self) -> bool:
        return self.mode == "dflash"

    @property
    def uses_block_draft_model(self) -> bool:
        return self.mode in BLOCK_SPEC_MODES

    @property
    def needs_target_layer_hidden(self) -> bool:
        return self.mode in TARGET_HIDDEN_SPEC_MODES

    @property
    def uses_attention_draft(self) -> bool:
        return self.mode in ATTENTION_SPEC_MODES

    @property
    def uses_no_attention_draft(self) -> bool:
        return self.mode in NO_ATTENTION_SPEC_MODES

    @property
    def uses_chained_draft_models(self) -> bool:
        return self.mode in VANILLA_SPEC_MODES

    @property
    def uses_recurrent_draft_model(self) -> bool:
        return self.mode in EAGLE_SPEC_MODES

    @property
    def draft_model_count(self) -> int:
        if not self.enabled:
            return 0
        return 1 if (self.uses_recurrent_draft_model or self.uses_block_draft_model) else self.step

    @property
    def needs_draft_vocab_mapping(self) -> bool:
        return self.is_eagle3

    def get_decode_graph_mtp_step(self, *, model_config: Mapping[str, Any], is_draft_model: bool) -> int:
        if (self.is_dflash or self.is_dspark) and is_draft_model:
            return int(model_config["block_size"]) - 1
        return self.step

    def validate(self) -> None:
        if not self.enabled:
            assert self.step == 0
            return

        assert self.mode in SPEC_MODES, f"unsupported speculative mode {self.mode}"
        if not self.uses_block_draft_model:
            assert self.step > 0
        else:
            assert self.step >= 0
        if self.is_dspark:
            assert self.dynamic_verify, "DSpark mode requires dynamic verify scheduling"
        if self.uses_chained_draft_models:
            assert self.draft_model_count == self.step
        else:
            assert self.draft_model_count == 1


def is_eagle3_draft_config(config: Mapping[str, Any]) -> bool:
    architectures = config.get("architectures", [])
    return config.get("model_type") == "llama" or any(
        architecture in ["Eagle3Speculator", "Qwen3Eagle3Model"] for architecture in architectures
    )


def is_dspark_draft_config(config: Mapping[str, Any]) -> bool:
    architectures = config.get("architectures", [])
    return any(architecture in DSPARK_FAMILY_ARCHITECTURES for architecture in architectures)


def is_qwen3_dflash_draft_config(config: Mapping[str, Any]) -> bool:
    architectures = config.get("architectures", [])
    return any(architecture in QWEN3_DFLASH_ARCHITECTURES for architecture in architectures)


def is_qwen3_dspark_draft_config(config: Mapping[str, Any]) -> bool:
    architectures = config.get("architectures", [])
    return any(architecture in QWEN3_DSPARK_ARCHITECTURES for architecture in architectures)


def is_gemma4_dspark_draft_config(config: Mapping[str, Any]) -> bool:
    architectures = config.get("architectures", [])
    return any(architecture in GEMMA4_DSPARK_ARCHITECTURES for architecture in architectures)


def validate_dspark_family_draft_config(
    config: Mapping[str, Any],
    *,
    require_confidence_head: bool = False,
) -> None:
    """Validate DFlash/DSpark checkpoint fields consumed by LightLLM serving."""

    assert is_dspark_draft_config(config), f"unsupported DFlash/DSpark architecture: {config.get('architectures')}"

    block_size = int(config.get("block_size", 0))
    assert block_size > 0, "DFlash/DSpark draft config must provide positive block_size"

    target_layer_ids = config.get("target_layer_ids")
    assert (
        isinstance(target_layer_ids, (list, tuple)) and len(target_layer_ids) > 0
    ), "DFlash/DSpark draft config must provide non-empty target_layer_ids"
    previous_layer_id = None
    for layer_id in target_layer_ids:
        layer_id = int(layer_id)
        assert layer_id >= 0, (
            "LightLLM DFlash/DSpark serving expects decoder-layer target_layer_ids; "
            "embedding-output layer_id=-1 is not supported"
        )
        assert (
            previous_layer_id is None or layer_id > previous_layer_id
        ), "DFlash/DSpark target_layer_ids must be strictly increasing"
        previous_layer_id = layer_id

    assert "mask_token_id" in config, "DFlash/DSpark draft config must provide mask_token_id"
    assert int(config["mask_token_id"]) >= 0, "DFlash/DSpark mask_token_id must be non-negative"

    markov_rank = int(config.get("markov_rank", 0))
    assert markov_rank >= 0, f"DFlash/DSpark markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        markov_head_type = str(config.get("markov_head_type", "")).lower()
        assert (
            markov_head_type in DSPARK_MARKOV_HEAD_TYPES
        ), f"unsupported DFlash/DSpark markov_head_type {markov_head_type!r}"

    enable_confidence_head = bool(config.get("enable_confidence_head", False))
    if require_confidence_head:
        assert enable_confidence_head, "DSpark dynamic scheduling requires enable_confidence_head=true"
    if enable_confidence_head:
        assert (
            "confidence_head_with_markov" in config
        ), "confidence_head_with_markov must be provided when enable_confidence_head is true"
        if bool(config.get("confidence_head_with_markov", False)):
            assert markov_rank > 0, "confidence_head_with_markov requires markov_rank > 0"
    return


def get_dspark_family_block_size(
    config: Mapping[str, Any],
    *,
    require_confidence_head: bool = False,
) -> int:
    validate_dspark_family_draft_config(
        config,
        require_confidence_head=require_confidence_head,
    )
    return int(config["block_size"])
