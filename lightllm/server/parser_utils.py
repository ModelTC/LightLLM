# Copyright 2025 ModelTC Team
# Copyright 2023-2025 SGLang Team
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

import re
from typing import Optional, Sequence, Tuple


def prefix_hold(text: str, markers: Sequence[str]) -> Tuple[str, str]:
    """Split off the longest suffix that could become a complete marker."""
    max_length = min(len(text), max((len(marker) - 1 for marker in markers), default=0))
    for length in range(max_length, 0, -1):
        suffix = text[-length:]
        if any(len(suffix) < len(marker) and marker.startswith(suffix) for marker in markers):
            return text[:-length], suffix
    return text, ""


class GlmToolCallScanner:
    """Find the outer GLM tool-call end while treating argument values as literal text."""

    _MARKERS = ("<arg_value>", "</arg_value>", "</tool_call>")
    _MARKER_PATTERN = re.compile(r"</?arg_value>|</tool_call>")

    def __init__(self):
        self._in_arg_value = False
        self._pending_marker = ""

    def find_end(self, new_text: str) -> Optional[int]:
        """Return the closing tag's end offset in this chunk, or None if incomplete.

        Callers may forward every character immediately. The retained marker
        prefix is only for recognizing a delimiter split across chunks.
        """
        pending_length = len(self._pending_marker)
        text = self._pending_marker + new_text
        self._pending_marker = ""
        offset = 0

        for match in self._MARKER_PATTERN.finditer(text):
            offset = match.end()
            marker = match.group()
            if self._in_arg_value:
                if marker == "</arg_value>":
                    self._in_arg_value = False
            elif marker == "<arg_value>":
                self._in_arg_value = True
            elif marker == "</tool_call>":
                return offset - pending_length

        _, self._pending_marker = prefix_hold(text[offset:], self._MARKERS)
        return None
