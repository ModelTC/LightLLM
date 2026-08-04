from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional


VANILLA_SPEC_MODES = frozenset({"vanilla_with_att", "vanilla_no_att", "qwen3next_vanilla"})
EAGLE_SPEC_MODES = frozenset({"eagle_with_att", "eagle_no_att", "eagle3", "qwen3next_eagle"})
BLOCK_SPEC_MODES = frozenset({"dspark", "dflash"})
SPEC_MODES = VANILLA_SPEC_MODES | EAGLE_SPEC_MODES | BLOCK_SPEC_MODES

ATTENTION_SPEC_MODES = frozenset({"vanilla_with_att", "eagle_with_att", "eagle3", "dspark", "dflash"})
NO_ATTENTION_SPEC_MODES = frozenset({"vanilla_no_att", "eagle_no_att", "qwen3next_vanilla", "qwen3next_eagle"})
TARGET_HIDDEN_SPEC_MODES = frozenset({"eagle3", "dspark", "dflash"})
QWEN3_DFLASH_ARCHITECTURES = frozenset({"Qwen3DFlashModel", "Qwen3DSparkModel"})
QWEN3_5_DFLASH_ARCHITECTURES = frozenset({"Qwen3_5DFlashModel"})
QWEN3_DSPARK_ARCHITECTURES = frozenset({"Qwen3DSparkModel"})
GEMMA4_DSPARK_ARCHITECTURES = frozenset({"Gemma4DSparkModel"})
DSPARK_FAMILY_ARCHITECTURES = QWEN3_DFLASH_ARCHITECTURES | QWEN3_5_DFLASH_ARCHITECTURES | GEMMA4_DSPARK_ARCHITECTURES
DSPARK_MARKOV_HEAD_TYPES = frozenset({"vanilla", "gated", "rnn"})
SPECULATIVE_CONFIG_SECTIONS = ("dflash_config", "dspark_config", "draft_config", "speculative_config", "mtp_config")
SPECULATIVE_DRAFT_CONFIG_KEYS = frozenset(
    {
        "block_size",
        "target_layer_ids",
        "mask_token_id",
        "markov_rank",
        "markov_head_type",
        "enable_confidence_head",
        "confidence_head_with_markov",
    }
)


@dataclass(frozen=True)
class BlockDraftLayout:
    """Runtime layout of a non-causal block draft checkpoint.

    ``query_block_size`` is the number of logits rows emitted for one anchor,
    while ``proposal_output_start`` identifies the first row that represents a
    draft token. Keeping both values explicit lets serving support checkpoints
    whose query block includes a leading bonus row without teaching generic
    scheduling or proposer code about a particular model architecture.
    """

    query_block_size: int
    proposal_output_start: int

    @property
    def draft_step(self) -> int:
        return self.query_block_size - self.proposal_output_start

    def resolve_draft_step(self, configured_step: int) -> int:
        """Use a positive configured step up to the checkpoint's proposal capacity."""
        configured_step = int(configured_step)
        return configured_step if 0 < configured_step <= self.draft_step else self.draft_step


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
        if self.is_eagle3 and self.dynamic_verify:
            # Dynamic Eagle3 physically compacts target rows and recurrent
            # draft rows, so graph shapes must be available at unit batch
            # granularity instead of only at multiples of the exposed depth.
            return 0
        return self.step

    def get_decode_graph_warmup_mtp_step(self, *, model_config: Mapping[str, Any], is_draft_model: bool) -> int:
        if self.is_eagle3 and self.dynamic_verify:
            # Preserve representative MTP indices/shared-group metadata while
            # retaining unit-granularity graph shape capture.
            return min(3, self.step)
        return self.get_decode_graph_mtp_step(
            model_config=model_config,
            is_draft_model=is_draft_model,
        )

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


def is_qwen3_5_dflash_draft_config(config: Mapping[str, Any]) -> bool:
    architectures = config.get("architectures", [])
    return any(architecture in QWEN3_5_DFLASH_ARCHITECTURES for architecture in architectures)


def is_qwen3_dspark_draft_config(config: Mapping[str, Any]) -> bool:
    architectures = config.get("architectures", [])
    return any(architecture in QWEN3_DSPARK_ARCHITECTURES for architecture in architectures)


def is_gemma4_dspark_draft_config(config: Mapping[str, Any]) -> bool:
    architectures = config.get("architectures", [])
    return any(architecture in GEMMA4_DSPARK_ARCHITECTURES for architecture in architectures)


def normalize_speculative_draft_config(config: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Normalize supported speculative draft config layouts in place.

    LightLLM model code generally normalizes nested checkpoint config once at
    load time, then downstream code reads a flat `network_config`. This helper
    applies the same pattern to speculative draft checkpoints whose shared
    fields may live under sections such as `dflash_config`.
    """

    for section in SPECULATIVE_CONFIG_SECTIONS:
        nested_config = config.get(section)
        if not isinstance(nested_config, Mapping):
            continue
        for key in SPECULATIVE_DRAFT_CONFIG_KEYS:
            if key not in config and key in nested_config:
                config[key] = nested_config[key]
    return config


def validate_dspark_family_draft_config(
    config: MutableMapping[str, Any],
    *,
    require_confidence_head: bool = False,
) -> None:
    """Validate DFlash/DSpark checkpoint fields consumed by LightLLM serving."""

    assert is_dspark_draft_config(config), f"unsupported DFlash/DSpark architecture: {config.get('architectures')}"

    normalize_speculative_draft_config(config)

    block_size = int(config.get("block_size", 0))
    assert block_size > 0, "DFlash/DSpark draft config must provide positive block_size"

    target_layer_ids = config.get("target_layer_ids")
    assert (
        isinstance(target_layer_ids, (list, tuple)) and len(target_layer_ids) > 0
    ), "DFlash/DSpark draft config must provide non-empty target_layer_ids"
    previous_layer_id = None
    for raw_layer_id in target_layer_ids:
        layer_id = int(raw_layer_id)
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


def get_block_draft_layout(
    config: MutableMapping[str, Any],
    *,
    mode: str,
    require_confidence_head: bool = False,
) -> BlockDraftLayout:
    """Resolve the generic query/proposal layout of a block draft checkpoint.

    Proposal rows start at zero by default. Architectures with a different
    upstream block contract are registered explicitly.
    """

    assert mode in BLOCK_SPEC_MODES, f"block draft layout is not defined for mode {mode!r}"
    validate_dspark_family_draft_config(
        config,
        require_confidence_head=require_confidence_head,
    )

    query_block_size = int(config["block_size"])
    # The Z-Lab Qwen3.5 DFlash checkpoint defines block_size as the full
    # query block: row 0 is the accepted/bonus query and proposals start at 1.
    proposal_output_start = 1 if is_qwen3_5_dflash_draft_config(config) else 0

    assert 0 <= proposal_output_start < query_block_size, (
        "block draft proposal_output_start must be within the query block: "
        f"start={proposal_output_start}, block_size={query_block_size}"
    )
    return BlockDraftLayout(
        query_block_size=query_block_size,
        proposal_output_start=proposal_output_start,
    )
