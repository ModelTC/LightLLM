# SPDX-License-Identifier: Apache-2.0
"""ReplaySSM kernel exports, in lifecycle order.

decode writes history for the current window.
cursor records how many of those tokens were accepted.
fold applies that accepted prefix onto the canonical SSM.
"""

from .replayssm_decode import (
    _decode_replay,
    _gdn_precomputed_window,
    _project_qk,
    _replay_each_token,
)
from .replayssm_delta import _delta_rule_step
from .replayssm_cursor import _advance_cursor
from .replayssm_fold import _fold_prefix

__all__ = [
    "_decode_replay",
    "_project_qk",
    "_gdn_precomputed_window",
    "_replay_each_token",
    "_delta_rule_step",
    "_advance_cursor",
    "_fold_prefix",
]
