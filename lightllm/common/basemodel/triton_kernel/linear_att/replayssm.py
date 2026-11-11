# SPDX-License-Identifier: Apache-2.0
"""ReplaySSM kernel exports, in lifecycle order.

decode writes history for the current window and does not move the cursor.
advance_cursor moves the cursor after the last layer; verify trims the rejected tail.
fold applies that accepted prefix onto the canonical SSM.
"""

from .replayssm_decode import (
    _decode_replay,
    _gdn_precomputed_window,
    _kda_precomputed_window,
    _project_qk,
    _replay_each_token,
)
from .replayssm_delta import _apply_accepted_prefix, _delta_rule_step
from .replayssm_cursor import _advance_cursor
from .replayssm_fold import _fold_prefix

__all__ = [
    "_decode_replay",
    "_project_qk",
    "_gdn_precomputed_window",
    "_kda_precomputed_window",
    "_replay_each_token",
    "_apply_accepted_prefix",
    "_delta_rule_step",
    "_advance_cursor",
    "_fold_prefix",
]
