from collections import deque

from .sampling_params import SamplingParams


class StopSequenceOutput:
    """Filter one choice's text while retaining every token's metadata and finish status.

    Run at the complete request boundary: PD nodes must send their original text
    so the master can handle stops spanning prefill/decode or segment boundaries.
    """

    def __init__(self, sampling_params: SamplingParams):
        self.stop_strings = sampling_params.stop_sequences.to_strings()
        self.include_stop = sampling_params.include_stop_str_in_output
        self.tail_length = max((len(stop) for stop in self.stop_strings), default=1) - 1
        self.buffer_length = 0 if self.include_stop else self.tail_length
        self.pending_tokens = deque()
        self.pending_text = ""
        self.emitted_tail = ""

    def process(self, sub_req_id, text, metadata, finish_status):
        self.pending_tokens.append((sub_req_id, text, metadata, finish_status))
        self.pending_text += text
        finished = finish_status.is_finished()
        output_end = len(self.pending_text) if finished else len(self.pending_text) - self.buffer_length

        if finish_status.is_stopped():
            search_text = self.emitted_tail + self.pending_text
            stop_index = len(search_text)
            matched_stop = ""
            # to_strings() orders longer stops first for ties at the same position.
            for stop in self.stop_strings:
                index = search_text.find(stop)
                if index != -1 and index < stop_index:
                    stop_index = index
                    matched_stop = stop
            if matched_stop:
                output_end = stop_index + (len(matched_stop) if self.include_stop else 0) - len(self.emitted_tail)

        while self.pending_tokens:
            sub_req_id, token_text, metadata, token_finish_status = self.pending_tokens[0]
            # Hold whole tokens so text stays associated with its original ID/logprob.
            if not finished and len(token_text) > output_end:
                break
            self.pending_tokens.popleft()
            self.pending_text = self.pending_text[len(token_text) :]
            visible_text = token_text[: max(0, output_end)]
            output_end -= len(token_text)
            if self.include_stop and self.tail_length:
                self.emitted_tail = (self.emitted_tail + token_text)[-self.tail_length :]
            yield sub_req_id, visible_text, metadata, token_finish_status
