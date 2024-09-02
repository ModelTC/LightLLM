# Adapted from outlines-dev/outlines/blob/main/outlines/fsm/regex.py
# of the outlines-dev/outlines GitHub repository.
#
# Copyright 2023 ModelTC Team
# Copyright 2023 Outlines Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Dict, Set
from outlines.fsm import regex
from outlines.fsm.regex import BetterFSM
from outlines.fsm.regex import reduced_vocabulary, create_fsm_index_end_to_end


def new_create_fsm_index_tokenizer(
    fsm: BetterFSM,
    tokenizer,
) -> Tuple[Dict[int, Dict[int, int]], Set[int]]:
    """Construct an FMS index from a tokenizer.

    This uses the end-to-end approach of `create_fsm_index_end_to_end`.

    .. warning::

        `fsm` needs to be deterministically ordered so that future caching makes sense.

    """
    vocabulary, empty_token_ids = reduced_vocabulary(tokenizer)

    states_to_token_subsets = create_fsm_index_end_to_end(fsm.fsm_info, vocabulary)

    for state in fsm.fsm_info.finals:
        subset = states_to_token_subsets.get(state)
        if subset is not None:
            for eos_token_id in tokenizer.eos_token_ids:

                subset.add((eos_token_id, state))

    # Convert to token-to-end-state maps
    states_to_token_subsets = {k: dict(v) for k, v in states_to_token_subsets.items()}

    return states_to_token_subsets, empty_token_ids


regex.create_fsm_index_tokenizer.__code__ = new_create_fsm_index_tokenizer.__code__
