from collections import deque

from lightllm.server.reasoning_parser import ReasoningStopState
from lightllm.utils.envs_utils import get_env_start_args, get_stop_in_reasoning


class StopSequenceBuffer:
    """Match stops and hold complete tokens until their text is safe to emit.

    Payloads travel unchanged with each token, including tokens whose visible
    text is trimmed to empty. PD Master uses the same buffer across segments.
    """

    def __init__(self, sampling_params, tokenizer, prompt_ids):
        self.stop_strings = sampling_params.stop_sequences.to_strings()
        self.include_stop = sampling_params.include_stop_str_in_output
        self.stop_prefixes = {stop[:length] for stop in self.stop_strings for length in range(1, len(stop))}
        self.max_prefix_length = max((len(prefix) for prefix in self.stop_prefixes), default=0)
        self.stop_token_sequences = [
            group.to_list()
            for group in sampling_params.stop_sequences.groups[: sampling_params.stop_sequences.size]
            if group.sequence_str_len == 0
        ]
        self.token_tail = deque(maxlen=max((len(ids) for ids in self.stop_token_sequences), default=1))
        self.reasoning_state = None
        if sampling_params.stop_sequences.size:
            args = get_env_start_args()
            if args.reasoning_parser and not get_stop_in_reasoning():
                self.reasoning_state = ReasoningStopState(
                    args.reasoning_parser,
                    tokenizer,
                    sampling_params._initial_reasoning_state,
                    prompt_ids,
                )
        self.output_strs = deque()
        self.pending_text = ""
        self.ready_tokens = deque()

    def append(self, token_id, text, payload, simulated=False):
        """Queue a token; return whether it matches a string or scoped token stop."""
        self.output_strs.append((text, payload))
        self.pending_text += text
        can_match = not simulated
        if can_match and self.reasoning_state is not None:
            can_match = self.reasoning_state.update(token_id)
        if not can_match:
            self.token_tail.clear()
            self.flush()
            return False

        stop_index = len(self.pending_text)
        matched_stop = None
        for stop in self.stop_strings:
            index = self.pending_text.find(stop)
            if index != -1 and index < stop_index:
                stop_index, matched_stop = index, stop
        if matched_stop is not None:
            self.flush(stop_index + (len(matched_stop) if self.include_stop else 0))
            return True

        if self.reasoning_state is not None and self.stop_token_sequences:
            self.token_tail.append(int(token_id))
            tail = list(self.token_tail)
            if any(tail[-len(ids) :] == ids for ids in self.stop_token_sequences):
                self.flush()
                return True

        # Only a suffix that is a stop prefix needs to wait for another token.
        prefix_length = next(
            (
                length
                for length in range(min(len(self.pending_text), self.max_prefix_length), 0, -1)
                if self.pending_text[-length:] in self.stop_prefixes
            ),
            0,
        )
        safe_length = len(self.pending_text) - prefix_length
        while self.output_strs and len(self.output_strs[0][0]) <= safe_length:
            token_text, payload = self.output_strs.popleft()
            self.ready_tokens.append((token_text, payload))
            safe_length -= len(token_text)
            self.pending_text = self.pending_text[len(token_text) :]
        return False

    def flush(self, visible_length=None):
        """Release an unmatched tail, or trim at a confirmed string stop."""
        if visible_length is None:
            visible_length = len(self.pending_text)
        while self.output_strs:
            text, payload = self.output_strs.popleft()
            self.ready_tokens.append((text[: max(0, visible_length)], payload))
            visible_length -= len(text)
        self.pending_text = ""
