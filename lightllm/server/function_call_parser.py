# Adaptive from SGlang [https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call_parser.py]
# Copyright 2025 ModelTC Team
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import orjson
import logging
import re
from abc import ABC, abstractmethod
from json import JSONDecodeError, JSONDecoder
from json.decoder import WHITESPACE
from typing import Any, Dict, List, Optional, Tuple, Type

import partial_json_parser
from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow
from pydantic import BaseModel, Field

from .api_models import Tool

logger = logging.getLogger(__name__)

TOOLS_TAG_LIST = ["<|plugin|>", "<function=", "<tool_call>", "<|python_tag|>", "[TOOL_CALLS]", "<｜tool▁calls▁begin｜>"]


class ToolCallItem(BaseModel):
    """Simple encapsulation of the parsed ToolCall result for easier usage in streaming contexts."""

    tool_index: int
    name: Optional[str] = None
    parameters: str  # JSON string


class StreamingParseResult:
    """Result of streaming incremental parsing."""

    def __init__(self, normal_text: str = "", calls: Optional[List[ToolCallItem]] = None):
        self.normal_text = normal_text
        self.calls = calls or []


def _find_common_prefix(s1: str, s2: str) -> str:
    prefix = ""
    min_length = min(len(s1), len(s2))
    for i in range(0, min_length):
        if s1[i] == s2[i]:
            prefix += s1[i]
        else:
            break
    return prefix


def _partial_json_loads(input_str: str, flags: Allow) -> Tuple[Any, int]:
    """
    Parse incomplete or partial JSON strings commonly encountered during streaming.

    Args:
        input_str (str): The potentially incomplete JSON string to parse.
        flags (Allow): Bitwise flags controlling what types of partial data are allowed.
            Common flags include:
            - Allow.STR: Allow partial strings (e.g., '"hello wo' -> 'hello wo')
            - Allow.OBJ: Allow partial objects (e.g., '{"key":' -> {'key': None})
            - Allow.ARR: Allow partial arrays (e.g., '[1, 2,' -> [1, 2])
            - Allow.ALL: Allow all types of partial data

    Returns:
        Tuple[Any, int]: A tuple containing:
            - parsed_object: The Python object parsed from the JSON
            - consumed_length: Number of characters consumed from input_str
    """
    try:
        return (partial_json_parser.loads(input_str, flags), len(input_str))
    except (JSONDecodeError, IndexError) as e:
        msg = getattr(e, "msg", str(e))
        if "Extra data" in msg or "pop from empty list" in msg:
            start = WHITESPACE.match(input_str, 0).end()
            obj, end = JSONDecoder().raw_decode(input_str, start)
            return obj, end
        raise


def _is_complete_json(input_str: str) -> bool:
    try:
        orjson.loads(input_str)
        return True
    except JSONDecodeError:
        return False


class BaseFormatDetector(ABC):
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    def __init__(self):
        # Streaming state management
        # Buffer for accumulating incomplete patterns that arrive across multiple streaming chunks
        self._buffer = ""
        # Stores complete tool call info (name and arguments) for each tool being parsed.
        # Used by serving layer for completion handling when streaming ends.
        # Format: [{"name": str, "arguments": dict}, ...]
        self.prev_tool_call_arr: List[Dict] = []
        # Index of currently streaming tool call. Starts at -1 (no active tool),
        # increments as each tool completes. Tracks which tool's arguments are streaming.
        self.current_tool_id: int = -1
        # Flag for whether current tool's name has been sent to client.
        # Tool names sent first with empty parameters, then arguments stream incrementally.
        self.current_tool_name_sent: bool = False
        # Tracks raw JSON string content streamed to client for each tool's arguments.
        # Critical for serving layer to calculate remaining content when streaming ends.
        # Each index corresponds to a tool_id. Example: ['{"location": "San Francisco"', '{"temp": 72']
        self.streamed_args_for_tool: List[str] = []

        # Token configuration (override in subclasses)
        self.bot_token = ""
        self.eot_token = ""
        self.tool_call_separator = ", "

    def _get_tool_indices(self, tools: List[Tool]) -> Dict[str, int]:
        """
        Get a mapping of tool names to their indices in the tools list.

        This utility method creates a dictionary mapping function names to their
        indices in the tools list, which is commonly needed for tool validation
        and ToolCallItem creation.

        Args:
            tools: List of available tools

        Returns:
            Dictionary mapping tool names to their indices
        """
        return {tool.function.name: i for i, tool in enumerate(tools) if tool.function.name}

    def parse_base_json(self, action: Any, tools: List[Tool]) -> List[ToolCallItem]:
        tool_indices = self._get_tool_indices(tools)
        if not isinstance(action, list):
            action = [action]

        results = []
        for act in action:
            name = act.get("name")
            if name and name in tool_indices:
                results.append(
                    ToolCallItem(
                        tool_index=-1,  # Caller should update this based on the actual tools array called
                        name=name,
                        parameters=json.dumps(
                            act.get("parameters") or act.get("arguments", {}),
                            ensure_ascii=False,
                        ),
                    )
                )
            else:
                logger.warning(f"Model attempted to call undefined function: {name}")

        return results

    @abstractmethod
    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Parses the text in one go. Returns success=True if the format matches, otherwise False.
        Note that leftover_text here represents "content that this parser will not consume further".
        """
        action = orjson.loads(text)
        return StreamingParseResult(calls=self.parse_base_json(action, tools))

    def _ends_with_partial_token(self, buffer: str, bot_token: str) -> int:
        """
        Check if buffer ends with a partial bot_token.
        Return the length of the partial bot_token.

        For some format, the bot_token is not a token in model's vocabulary, such as
        `[TOOL_CALLS] [` in Mistral.
        """
        for i in range(1, min(len(buffer) + 1, len(bot_token))):
            if bot_token.startswith(buffer[-i:]):
                return i
        return 0

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Streaming incremental parsing with tool validation.

        This base implementation works best with formats where:
        1. bot_token is followed immediately by JSON (e.g., bot_token + JSON_array)
        2. JSON can be parsed incrementally using partial_json_loads
        3. Multiple tool calls are separated by "; " or ", "

        Examples of incompatible formats (need custom implementation, may reuse some logic from this class):
        - Each tool call is wrapped in a separate block: See Qwen25Detector
        - Multiple separate blocks: [TOOL_CALLS] [...] \n [TOOL_CALLS] [...]
        - Tool call is Pythonic style

        For incompatible formats, detectors should override this method with custom logic.
        """
        # Append new text to buffer
        self._buffer += new_text
        current_text = self._buffer

        # The current_text has tool_call if it is the start of a new tool call sequence
        # or it is the start of a new tool call after a tool call separator, when there is a previous tool call
        if not (
            self.has_tool_call(current_text)
            or (self.current_tool_id > 0 and current_text.startswith(self.tool_call_separator))
        ):
            # Only clear buffer if we're sure no tool call is starting
            if not self._ends_with_partial_token(self._buffer, self.bot_token):
                normal_text = self._buffer
                self._buffer = ""
                if self.eot_token in normal_text:
                    normal_text = normal_text.replace(self.eot_token, "")
                return StreamingParseResult(normal_text=normal_text)
            else:
                # Might be partial bot_token, keep buffering
                return StreamingParseResult()

        # Build tool indices if not already built
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

        try:
            try:
                tool_call_pos = current_text.find(self.bot_token)
                if tool_call_pos != -1:
                    start_idx = tool_call_pos + len(self.bot_token)
                elif self.current_tool_id > 0 and current_text.startswith(self.tool_call_separator):
                    start_idx = len(self.tool_call_separator)
                else:
                    start_idx = 0

                if start_idx >= len(current_text):
                    return StreamingParseResult()

                (obj, end_idx) = _partial_json_loads(current_text[start_idx:], flags)

                is_current_complete = _is_complete_json(current_text[start_idx : start_idx + end_idx])

                # Validate tool name if present
                if "name" in obj and obj["name"] not in self._tool_indices:
                    # Invalid tool name - reset state
                    self._buffer = ""
                    self.current_tool_id = -1
                    self.current_tool_name_sent = False
                    if self.streamed_args_for_tool:
                        self.streamed_args_for_tool.pop()
                    return StreamingParseResult()

                # Handle parameters/arguments consistency
                # NOTE: we assume here that the obj is always partial of a single tool call
                if "parameters" in obj:
                    assert "arguments" not in obj, "model generated both parameters and arguments"
                    obj["arguments"] = obj["parameters"]

                current_tool_call = obj

            except MalformedJSON:
                return StreamingParseResult()

            if not current_tool_call:
                return StreamingParseResult()

            # Case 1: Handle tool name streaming
            # This happens when we encounter a tool but haven't sent its name yet
            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")

                if function_name and function_name in self._tool_indices:
                    # If this is a new tool (current_tool_id was -1), initialize it
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                        self.streamed_args_for_tool.append("")
                    # If this is a subsequent tool, ensure streamed_args_for_tool is large enough
                    elif self.current_tool_id >= len(self.streamed_args_for_tool):
                        while len(self.streamed_args_for_tool) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")

                    # Send the tool name with empty parameters
                    res = StreamingParseResult(
                        calls=[
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=function_name,
                                parameters="",
                            )
                        ],
                    )
                    self.current_tool_name_sent = True
                else:
                    res = StreamingParseResult()

            # Case 2: Handle streaming arguments
            # This happens when we've already sent the tool name and now need to stream arguments incrementally
            else:
                cur_arguments = current_tool_call.get("arguments")
                res = StreamingParseResult()

                if cur_arguments:
                    # Calculate how much of the arguments we've already streamed
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments)
                    prev_arguments = None
                    if self.current_tool_id < len(self.prev_tool_call_arr):
                        prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get("arguments")

                    argument_diff = None

                    # If the current tool's JSON is complete, send all remaining arguments
                    if is_current_complete:
                        argument_diff = cur_args_json[sent:]
                        completing_tool_id = self.current_tool_id  # Save the ID of the tool that's completing

                        # Only remove the processed portion, keep unprocessed content
                        self._buffer = current_text[start_idx + end_idx :]

                        if self.current_tool_id < len(self.prev_tool_call_arr):
                            self.prev_tool_call_arr[self.current_tool_id].clear()
                        self.current_tool_name_sent = False
                        self.streamed_args_for_tool[self.current_tool_id] = ""
                        self.current_tool_id += 1

                    # If the tool is still being parsed, send incremental changes
                    elif prev_arguments:
                        prev_args_json = json.dumps(prev_arguments)
                        if cur_args_json != prev_args_json:
                            prefix = _find_common_prefix(prev_args_json, cur_args_json)
                            argument_diff = prefix[sent:]

                    # Send the argument diff if there's something new
                    if argument_diff is not None:
                        # Use the correct tool_index: completing_tool_id for completed tools,
                        # current_tool_id for ongoing
                        tool_index_to_use = completing_tool_id if is_current_complete else self.current_tool_id
                        res = StreamingParseResult(
                            calls=[
                                ToolCallItem(
                                    tool_index=tool_index_to_use,
                                    parameters=argument_diff,
                                )
                            ],
                        )
                        if not is_current_complete:
                            self.streamed_args_for_tool[self.current_tool_id] += argument_diff

            # Update prev_tool_call_arr with current state
            if self.current_tool_id >= 0:
                # Ensure prev_tool_call_arr is large enough
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                self.prev_tool_call_arr[self.current_tool_id] = current_tool_call

            return res

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult()


class Qwen25Detector(BaseFormatDetector):
    """
    Detector for Qwen 2.5 and Qwen 3 model function call format.

    Format Structure:
    ```
    <tool_call>\n{"name":"func1", "arguments":{...}}\n
    </tool_call>\n<tool_call>\n{"name":"func2", "arguments":{...}}\n</tool_call>
    ```

    Key Components:
    - Tool Call Tags: `<tool_call>` and `</tool_call>` wrap each individual call
    - Function Call Object: JSON object with "name" and "arguments" fields

    Reference: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct?chat_template=default
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"
        self.tool_call_separator = "\n"
        self._normal_text_buffer = ""  # Buffer for handling partial end tokens

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Qwen 2.5 format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        # Find all <tool_call>\n...\n</tool_call> blocks
        pattern = rf"{re.escape(self.bot_token)}(.*?){re.escape(self.eot_token)}"
        match_result_list = re.findall(pattern, text, re.DOTALL)
        calls = []
        for match_result in match_result_list:
            try:
                parsed_call = json.loads(match_result.strip())
                calls.extend(self.parse_base_json(parsed_call, tools))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON part: {match_result}, JSON parse error: {str(e)}")
                continue
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Streaming incremental parsing for Qwen 2.5 tool calls.
        Uses base class implementation with buffering to handle partial end tokens.
        """
        result = super().parse_streaming_increment(new_text, tools)

        # Handle partial end tokens that are streamed character by character
        if result.normal_text:
            self._normal_text_buffer += result.normal_text

            # Check if buffer contains complete end token (without leading newline)
            end_token_without_newline = self.eot_token  # "</tool_call>"
            if end_token_without_newline in self._normal_text_buffer:
                cleaned_text = self._normal_text_buffer.replace(end_token_without_newline, "")
                self._normal_text_buffer = ""
                result.normal_text = cleaned_text
            else:
                # Check if buffer might contain partial end token at the end
                partial_match_len = self._ends_with_partial_token(self._normal_text_buffer, end_token_without_newline)

                if partial_match_len:
                    # Keep potential partial match in buffer, return the rest
                    result.normal_text = self._normal_text_buffer[:-partial_match_len]
                    self._normal_text_buffer = self._normal_text_buffer[-partial_match_len:]
                else:
                    # No partial match, return all buffered text
                    result.normal_text = self._normal_text_buffer
                    self._normal_text_buffer = ""

        return result


class MistralDetector(BaseFormatDetector):
    """
    Detector for Mistral model function call format.

    The Mistral format uses a simple bracket-delimited structure with JSON arrays
    containing function call objects.

    Format Structure:
    ```
    [TOOL_CALLS] [{"name": "function_name", "arguments": {json_args}}, ...]
    ```

    Reference: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3?chat_template=default
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self.bot_token = "[TOOL_CALLS] ["
        self.eot_token = "]"
        self.tool_call_regex = re.compile(r"\[{.*}\]", re.DOTALL)
        self.tool_call_separator = ", "

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Mistral format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text

        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        # Extract the JSON array part from [TOOL_CALLS] [...]
        # Use bracket counting to properly handle nested brackets in JSON content
        json_array_str = self._extract_json_array(text)
        if not json_array_str:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        calls = []
        try:
            function_call_arr = json.loads(json_array_str)
            # Handle both single object and array of objects
            if not isinstance(function_call_arr, list):
                function_call_arr = [function_call_arr]
            calls = self.parse_base_json(function_call_arr, tools)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON part: {json_array_str}, JSON parse error: {str(e)}")

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _extract_json_array(self, text: str) -> str:
        """
        Extract the JSON array part using bracket counting to handle nested brackets.

        :param text: The complete text containing [TOOL_CALLS] [...]
        :return: The JSON array string or None if not found
        """
        start_idx = text.find(self.bot_token)
        if start_idx == -1:
            return None

        # Start from the opening bracket after [TOOL_CALLS]
        json_start = start_idx + len(self.bot_token) - 1  # -1 to include the opening bracket
        bracket_count = 0
        in_string = False
        escape_next = False

        for i in range(json_start, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == "[":
                    bracket_count += 1
                elif char == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        return text[json_start : i + 1]

        return None


class Llama32Detector(BaseFormatDetector):
    """
    Detector for Llama 3.2 models with json tool call format.

    Format Structure:
    ```
    <python_tag>{"name":"xxx", "arguments":{...}}
    ```
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<|python_tag|>"
        # NOTE: technically Llama3.2 doesn't support well with parallel tool calls
        # They need specific prompt engineering to support parallel tool calls
        # Here we use ';' as the separator, which might have compatibility issues
        # if users define to use a different separator in their prompt
        self.tool_call_separator = ";"

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Llama 3.2 format tool call."""
        # depending on the prompt format the Llama model may or may not
        # prefix the output with the <|python_tag|> token
        return "<|python_tag|>" in text or text.startswith("{")

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Parse function calls from text, handling multiple JSON objects."""
        if "<|python_tag|>" not in text and not text.startswith("{"):
            return StreamingParseResult(normal_text=text, calls=[])

        if "<|python_tag|>" in text:
            normal_text, action_text = text.split("<|python_tag|>", maxsplit=1)
        else:
            normal_text, action_text = "", text

        decoder = json.JSONDecoder()
        idx = 0
        safe_idx = idx  # the index of the last valid JSON object
        all_actions = []
        action_text_len = len(action_text)
        while idx < action_text_len:
            try:
                obj, end = decoder.raw_decode(action_text[idx:])
                all_actions.append(obj)
                idx += end + len(self.tool_call_separator)
                safe_idx = idx
            except json.JSONDecodeError as e:
                # Find where next `{"name"` appears and try again
                logger.warning(f"Failed to parse JSON part: {action_text[idx:]}, JSON parse error: {str(e)}")
                next_obj_start = action_text.find('{"name":', idx + 1)
                if next_obj_start == -1:
                    break
                idx = next_obj_start
                continue

        # Only process if we found valid JSON objects
        calls = self.parse_base_json(all_actions, tools) if all_actions else []
        # Use safe_idx to avoid idx containing the last part of an invalid JSON object
        trailing_text = action_text[safe_idx:].strip() if safe_idx < action_text_len else ""
        return StreamingParseResult(normal_text=normal_text + trailing_text, calls=calls)


class KimiK2Detector(BaseFormatDetector):
    """
    Detector for Kimi K2 model function call format.

    Format Structure:
    ```
    <|tool_calls_section_begin|>
    <|tool_call_begin|>functions.{func_name}:{index}<|tool_call_argument_begin|>{json_args}<|tool_call_end|>
    <|tool_calls_section_end|>
    ```

    Reference: https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/docs/tool_call_guidance.md
    """

    def __init__(self):
        super().__init__()

        self.bot_token: str = "<|tool_calls_section_begin|>"
        self.eot_token: str = "<|tool_calls_section_end|>"

        self.tool_call_start_token: str = "<|tool_call_begin|>"
        self.tool_call_end_token: str = "<|tool_call_end|>"

        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*?\})\s*<\|tool_call_end\|>"  # noqa
        )

        self.stream_tool_call_portion_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*)"  # noqa
        )

        self._last_arguments = ""

        # Robust parser for ids like "functions.search:0" or fallback "search:0"
        self.tool_call_id_regex = re.compile(r"^(?:functions\.)?(?P<name>[\w\.]+):(?P<index>\d+)$")

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a KimiK2 format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=text, calls=[])
        try:
            # there are two possible captures - between tags, or between a
            # tag and end-of-string so the result of
            # findall is an array of tuples where one is a function call and
            # the other is None
            function_call_tuples = self.tool_call_regex.findall(text)

            logger.debug("function_call_tuples: %s", function_call_tuples)

            tool_calls = []
            for match in function_call_tuples:
                function_id, function_args = match
                m = self.tool_call_id_regex.match(function_id)
                if not m:
                    logger.warning("Unexpected tool_call_id format: %s", function_id)
                    continue
                function_name = m.group("name")
                function_idx = int(m.group("index"))

                logger.info(f"function_name {function_name}")

                tool_calls.append(
                    ToolCallItem(
                        tool_index=function_idx,
                        name=function_name,
                        parameters=function_args,
                    )
                )

            content = text[: text.find(self.bot_token)]
            return StreamingParseResult(normal_text=content, calls=tool_calls)

        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for KimiK2 format.
        """
        self._buffer += new_text
        current_text = self._buffer

        # Check if we have a tool call (either the start token or individual tool call)
        has_tool_call = self.bot_token in current_text or self.tool_call_start_token in current_text

        if not has_tool_call:
            self._buffer = ""
            for e_token in [self.eot_token, self.tool_call_end_token]:
                if e_token in new_text:
                    new_text = new_text.replace(e_token, "")
            return StreamingParseResult(normal_text=new_text)

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: list[ToolCallItem] = []
        try:
            match = self.stream_tool_call_portion_regex.search(current_text)
            if match:
                function_id = match.group("tool_call_id")
                function_args = match.group("function_arguments")

                m = self.tool_call_id_regex.match(function_id)
                if not m:
                    logger.warning("Unexpected tool_call_id format: %s", function_id)
                    return StreamingParseResult(normal_text="", calls=calls)
                function_name = m.group("name")

                # Initialize state if this is the first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]

                # Ensure we have enough entries in our tracking arrays
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                if not self.current_tool_name_sent:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=function_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True
                    # Store the tool call info for serving layer completions endpoint
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": function_name,
                        "arguments": {},
                    }
                else:
                    argument_diff = (
                        function_args[len(self._last_arguments) :]
                        if function_args.startswith(self._last_arguments)
                        else function_args
                    )

                    parsed_args_diff = argument_diff.split("<|tool_call_end|>", 1)[0]

                    if parsed_args_diff:

                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=None,
                                parameters=parsed_args_diff,
                            )
                        )
                        self._last_arguments += argument_diff
                        self.streamed_args_for_tool[self.current_tool_id] += parsed_args_diff

                    parsed_args = function_args.split("<|tool_call_end|>", 1)[0]
                    if _is_complete_json(parsed_args):
                        try:
                            parsed_args = json.loads(parsed_args)
                            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = parsed_args
                        except json.JSONDecodeError:
                            pass

                        # Find the end of the current tool call and remove only that part from buffer
                        tool_call_end_pattern = r"<\|tool_call_begin\|>.*?<\|tool_call_end\|>"
                        match = re.search(tool_call_end_pattern, current_text, re.DOTALL)
                        if match:
                            # Remove the completed tool call from buffer, keep any remaining content
                            self._buffer = current_text[match.end() :]
                        else:
                            self._buffer = ""

                        result = StreamingParseResult(normal_text="", calls=calls)
                        self.current_tool_id += 1
                        self._last_arguments = ""
                        self.current_tool_name_sent = False
                        return result

            return StreamingParseResult(normal_text="", calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)


class DeepSeekV31Detector(BaseFormatDetector):
    """
    Detector for DeepSeek V3 model function call format.

    The DeepSeek V3 format uses special Unicode tokens to delimit function calls
    with JSON code blocks for arguments.

    Format Structure:
    ```
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{function_name}<｜tool▁sep｜>{json_arguments}
    <｜tool▁calls▁end｜><｜end▁of▁sentence｜>
    ```
    Examples:
    ```
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>
    {"location": "Tokyo"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_current_weather
    <｜tool▁sep｜>{"location": "Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜tool▁calls▁begin｜>` and `<｜tool▁calls▁end｜>`
    - Individual Tool Call: Wrapped between `<｜tool▁call▁begin｜>` and `<｜tool▁call▁end｜>`
    - Function Declaration: `<｜tool▁call▁begin｜>{function_name}<｜tool▁sep｜>`
    - Arguments: JSON code block between `<｜tool▁sep｜>` and `<｜tool▁call▁end｜>`
    - Supports multiple tool calls

    Reference: https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3.1
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜tool▁calls▁begin｜>"
        self.eot_token = "<｜tool▁calls▁end｜>"
        self.func_call_regex = r"<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>"
        self.func_detail_regex = r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)<｜tool▁call▁end｜>"
        self._last_arguments = ""
        self.current_tool_id = -1

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])
        match_result_list = re.findall(self.func_call_regex, text, re.DOTALL)
        calls = []
        try:
            for match_result in match_result_list:
                # Get function name
                func_detail = re.search(self.func_detail_regex, match_result, re.DOTALL)
                func_name = func_detail.group(1)
                func_args = func_detail.group(2)
                func_args = json.loads(func_args)
                # construct match_result for parse_base_json
                match_result = {"name": func_name, "parameters": func_args}
                calls.extend(self.parse_base_json(match_result, tools))
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for DeepSeekV3 format.
        """
        self._buffer += new_text
        current_text = self._buffer

        # Check if we have a tool call (either the start token or individual tool call)
        has_tool_call = self.bot_token in current_text or "<｜tool▁call▁begin｜>" in current_text

        if not has_tool_call:
            self._buffer = ""
            for e_token in [self.eot_token, "<｜tool▁call▁end｜>"]:
                if e_token in new_text:
                    new_text = new_text.replace(e_token, "")
            return StreamingParseResult(normal_text=new_text)

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: list[ToolCallItem] = []
        try:
            partial_match = re.search(
                pattern=r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)<｜tool▁call▁end｜>",
                string=current_text,
                flags=re.DOTALL,
            )
            if partial_match:
                func_name = partial_match.group(1).strip()
                func_args_raw = partial_match.group(2).strip()

                # Initialize state if this is the first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]

                # Ensure we have enough entries in our tracking arrays
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                if not self.current_tool_name_sent:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True
                    # Store the tool call info for serving layer completions endpoint
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": {},
                    }
                else:
                    argument_diff = (
                        func_args_raw[len(self._last_arguments) :]
                        if func_args_raw.startswith(self._last_arguments)
                        else func_args_raw
                    )

                    if argument_diff:
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=None,
                                parameters=argument_diff,
                            )
                        )
                        self._last_arguments += argument_diff
                        self.streamed_args_for_tool[self.current_tool_id] += argument_diff

                    if _is_complete_json(func_args_raw):
                        # Update the stored arguments
                        try:
                            parsed_args = json.loads(func_args_raw)
                            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = parsed_args
                        except json.JSONDecodeError:
                            pass

                        # Find the end of the current tool call and remove only that part from buffer
                        tool_call_end_pattern = r"<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>"
                        match = re.search(tool_call_end_pattern, current_text, re.DOTALL)
                        if match:
                            # Remove the completed tool call from buffer, keep any remaining content
                            self._buffer = current_text[match.end() :]
                        else:
                            self._buffer = ""

                        result = StreamingParseResult(normal_text="", calls=calls)
                        self.current_tool_id += 1
                        self._last_arguments = ""
                        self.current_tool_name_sent = False
                        return result

            return StreamingParseResult(normal_text="", calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)


class DeepSeekV3Detector(BaseFormatDetector):
    """
    Detector for DeepSeek V3 model function call format.

    The DeepSeek V3 format uses special Unicode tokens to delimit function calls
    with JSON code blocks for arguments.

    Format Structure:
    ```
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>{function_name}
    \n```json\n{json_arguments}\n```<｜tool▁calls▁end｜><｜end▁of▁sentence｜>
    ```
    Examples:
    ```
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n
    ```json\n{"location": "Tokyo"}\n```<｜tool▁call▁end｜>\n<｜tool▁call▁begin｜>
    function<｜tool▁sep｜>get_current_weather\n```json\n{"location": "Paris"}\n```
    <｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜tool▁calls▁begin｜>` and `<｜tool▁calls▁end｜>`
    - Individual Tool Call: Wrapped between `<｜tool▁call▁begin｜>` and `<｜tool▁call▁end｜>`
    - Function Declaration: `function<｜tool▁sep｜>{function_name}`
    - Arguments: JSON code block between ````json` and ````
    - Supports multiple tool calls

    Reference: https://huggingface.co/deepseek-ai/DeepSeek-V3-0324?chat_template=default
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜tool▁calls▁begin｜>"
        self.eot_token = "<｜tool▁calls▁end｜>"
        self.func_call_regex = r"<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>"
        self.func_detail_regex = r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)\n```json\n(.*)\n```<｜tool▁call▁end｜>"
        self._last_arguments = ""
        self.current_tool_id = -1

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])
        match_result_list = re.findall(self.func_call_regex, text, re.DOTALL)
        calls = []
        try:
            for match_result in match_result_list:
                # Get function name
                func_detail = re.search(self.func_detail_regex, match_result, re.DOTALL)
                func_name = func_detail.group(2)
                func_args = func_detail.group(3)
                func_args = json.loads(func_args)
                # construct match_result for parse_base_json
                match_result = {"name": func_name, "parameters": func_args}
                calls.extend(self.parse_base_json(match_result, tools))
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for DeepSeekV3 format.
        """
        self._buffer += new_text
        current_text = self._buffer

        # Check if we have a tool call (either the start token or individual tool call)
        has_tool_call = self.bot_token in current_text or "<｜tool▁call▁begin｜>" in current_text

        if not has_tool_call:
            self._buffer = ""
            for e_token in [self.eot_token, "```", "<｜tool▁call▁end｜>"]:
                if e_token in new_text:
                    new_text = new_text.replace(e_token, "")
            return StreamingParseResult(normal_text=new_text)

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: list[ToolCallItem] = []
        try:
            partial_match = re.search(
                pattern=r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)\n```json\n(.*)\n```.*",
                string=current_text,
                flags=re.DOTALL,
            )
            if partial_match:
                func_name = partial_match.group(2).strip()
                func_args_raw = partial_match.group(3).strip()

                # Initialize state if this is the first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]

                # Ensure we have enough entries in our tracking arrays
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                if not self.current_tool_name_sent:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True
                    # Store the tool call info for serving layer completions endpoint
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": {},
                    }
                else:
                    argument_diff = (
                        func_args_raw[len(self._last_arguments) :]
                        if func_args_raw.startswith(self._last_arguments)
                        else func_args_raw
                    )

                    if argument_diff:
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=None,
                                parameters=argument_diff,
                            )
                        )
                        self._last_arguments += argument_diff
                        self.streamed_args_for_tool[self.current_tool_id] += argument_diff

                    if _is_complete_json(func_args_raw):
                        # Update the stored arguments
                        try:
                            parsed_args = json.loads(func_args_raw)
                            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = parsed_args
                        except json.JSONDecodeError:
                            pass

                        # Find the end of the current tool call and remove only that part from buffer
                        tool_call_end_pattern = r"<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>"
                        match = re.search(tool_call_end_pattern, current_text, re.DOTALL)
                        if match:
                            # Remove the completed tool call from buffer, keep any remaining content
                            self._buffer = current_text[match.end() :]
                        else:
                            self._buffer = ""

                        result = StreamingParseResult(normal_text="", calls=calls)
                        self.current_tool_id += 1
                        self._last_arguments = ""
                        self.current_tool_name_sent = False
                        return result

            return StreamingParseResult(normal_text="", calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)


class FunctionCallParser:
    """
    Parser for function/tool calls in model outputs.

    This class handles both streaming and non-streaming parsing of function calls using a detector.
    In streaming scenarios, each time new_text is received, it calls detector.parse_streaming_increment
    and returns the resulting normal_text and calls to the upper layer (or SSE).
    """

    ToolCallParserEnum: Dict[str, Type[BaseFormatDetector]] = {
        "deepseekv3": DeepSeekV3Detector,
        "deepseekv31": DeepSeekV31Detector,
        "kimi_k2": KimiK2Detector,
        "llama3": Llama32Detector,
        "mistral": MistralDetector,
        "qwen": Qwen25Detector,
        "qwen25": Qwen25Detector,
    }

    def __init__(self, tools: List[Tool], tool_call_parser: str):
        detector: Type[BaseFormatDetector] = None
        detector_class = self.ToolCallParserEnum.get(tool_call_parser)
        if detector_class:
            detector = detector_class()
        else:
            raise ValueError(f"Unsupported tool_call_parser: {tool_call_parser}")

        self.detector = detector
        self.tools = tools

    def has_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains a tool call in the format supported by this parser.
        This delegates to the detector's implementation.

        Args:
            text: The text to check for tool calls

        Returns:
            True if the text contains a tool call, False otherwise
        """
        if not self.tools:
            return False
        return self.detector.has_tool_call(text)

    def parse_non_stream(self, full_text: str) -> Tuple[str, list[ToolCallItem]]:
        """
        One-time parsing of the full text to extract tool calls.

        Args:
            full_text: The complete text to parse

        Returns:
            A tuple containing:
            - The remaining text after parsing that was not consumed by the detector (can be treated as normal text)
            - A list of tool calls parsed from the text
        """
        if not self.tools:
            return full_text, []
        parsed_result = self.detector.detect_and_parse(full_text, self.tools)
        tool_call_list = parsed_result.calls
        if tool_call_list:
            return parsed_result.normal_text, tool_call_list
        else:
            return full_text, []

    def parse_stream_chunk(self, chunk_text: str) -> Tuple[str, list[ToolCallItem]]:
        """
        Streaming incremental parsing of chunks of text as they arrive.

        Args:
            chunk_text: The new chunk of text to parse

        Returns:
            A tuple containing:
            - The normal text that should be displayed to the user
            - A list of tool calls parsed from the chunk
        """
        if not self.tools:
            return chunk_text, []
        final_normal_text = ""
        final_calls = []

        sp_result = self.detector.parse_streaming_increment(chunk_text, self.tools)
        if sp_result.normal_text:
            final_normal_text = sp_result.normal_text
        if sp_result.calls:
            final_calls.extend(sp_result.calls)
            final_normal_text = sp_result.normal_text

        return final_normal_text, final_calls
