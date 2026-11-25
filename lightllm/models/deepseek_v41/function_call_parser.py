from typing import List

from lightllm.server.api_models import Tool
from lightllm.server.function_call_parser import DeepSeekV32Detector, StreamingParseResult, ToolCallItem


class DeepSeekV41Detector(DeepSeekV32Detector):
    """V4.1 uses space-prefixed DSML tags inside a calls wrapper."""

    dsml_token = "｜DSML｜ "

    def __init__(self):
        super().__init__()
        self.bot_token = f"<{self.dsml_token}calls>"
        self.eot_token = f"</{self.dsml_token}calls>"

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        # Buffer a calls block so parameter additions never invalidate already emitted JSON.
        self._buffer += new_text
        normal_text, calls = "", []
        while self._buffer:
            start = self._buffer.find(self.bot_token)
            if start == -1:
                partial = self._ends_with_partial_token(self._buffer, self.bot_token)
                normal_text += self._buffer[:-partial] if partial else self._buffer
                self._buffer = self._buffer[-partial:] if partial else ""
                break
            normal_text += self._buffer[:start]
            self._buffer = self._buffer[start:]
            end = self._buffer.find(self.eot_token)
            if end == -1:
                match = self.partial_invoke_regex.search(self._buffer)
                if match and not self.current_tool_name_sent and match.group(1) in self._get_tool_indices(tools):
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                    calls.append(ToolCallItem(tool_index=self.current_tool_id, name=match.group(1), parameters=""))
                    self.current_tool_name_sent = True
                break
            end += len(self.eot_token)
            parsed = self.detect_and_parse(self._buffer[:end], tools)
            if self.current_tool_id == -1:
                self.current_tool_id = 0
            for index, item in enumerate(parsed.calls):
                item.tool_index = self.current_tool_id
                if index == 0 and self.current_tool_name_sent:
                    item.name = None
                calls.append(item)
                self.current_tool_id += 1
            self.current_tool_name_sent = False
            self._buffer = self._buffer[end:]
        return StreamingParseResult(normal_text=normal_text, calls=calls)
