from dataclasses import dataclass
from enum import Enum, auto
from typing import Type, Union


class FinalHiddenPolicy(Enum):
    NONE = auto()
    PRE_NORM = auto()
    POST_NORM = auto()


@dataclass(frozen=True)
class MtpPairConfig:
    draft_model_class: Union[Type, str]
    main_final_hidden_policy: FinalHiddenPolicy
    draft_final_hidden_policy: FinalHiddenPolicy
