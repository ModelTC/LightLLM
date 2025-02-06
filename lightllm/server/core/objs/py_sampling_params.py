"""Sampling parameters for text generation.
本文件中定义的 SamplingParams 类是与 sampling_params.py 文件中实现的 基于ctypes 实现的 SamplingParams 具有
相同的功能，主要用于便于转化的易用性，目前在约束输出部分有使用到。
"""
import os
from typing import List, Optional, Union, Tuple
from transformers import GenerationConfig
from lightllm.server.req_id_generator import MAX_BEST_OF


_SAMPLING_EPS = 1e-5
# 用环境变量控制是否进行输入惩罚的默认值
DEFAULT_INPUT_PENALTY = os.getenv("INPUT_PENALTY", "False").upper() in ["ON", "TRUE", "1"]


class SamplingParams:

    _do_sample: bool = False
    _presence_penalty: float = 0.0
    _frequency_penalty: float = 0.0
    _repetition_penalty: float = 1.0
    _temperature: float = 1.0
    _top_p: float = 1.0
    _top_k: int = -1  # -1 is for all
    _stop_sequences: Optional[Union[str, List[str], List[List[int]]]] = None

    def __init__(
        self,
        best_of: int = 1,
        n: int = None,  # number of results
        do_sample: bool = None,
        presence_penalty: float = None,
        frequency_penalty: float = None,
        repetition_penalty: float = None,
        exponential_decay_length_penalty: Tuple[int, float] = (1, 1.0),
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,  # -1 is for all
        ignore_eos: bool = False,
        max_new_tokens: int = 16,
        min_new_tokens: int = 1,
        stop_sequences: Optional[Union[str, List[str], List[List[int]]]] = None,  # 停止句子条件
        skip_special_tokens: bool = True,  # whether to skip special tokens when decoding
        add_special_tokens: bool = True,  # whether to add special tokens when encoding
        add_spaces_between_special_tokens: bool = True,  # whether to add spaces between special tokens when decoding
        print_eos_token: bool = False,  # eos_id will be always ignored except the value is set to True
        # Whether to count input tokens for presence_penalty, frequency_penalty and repetition_penalty
        input_penalty: bool = DEFAULT_INPUT_PENALTY,
        regular_constraint: Optional[str] = None,  # Regular expressions constrain the output.
        # If provided, the engine will construct a logits,
        # processor which only retains scores for the given token ids. Defaults to None.
        # allowed_token_ids only can be used in "--simple_constraint_mode" started server.
        allowed_token_ids: Optional[List[int]] = None,
        # p d mode used params
        group_request_id: Optional[int] = None,
        # move kv to deocde node, only used in pd mode
        move_kv_to_decode_node: Optional[dict] = None,
        # suggest dp index, deepseekv2 dp mode, use to suggest used dp_index
        suggested_dp_index: Optional[int] = None,
    ) -> None:
        self.best_of = best_of
        self.n = n
        self.do_sample = do_sample if do_sample is not None else SamplingParams._do_sample
        self.presence_penalty = presence_penalty if presence_penalty is not None else SamplingParams._presence_penalty
        self.frequency_penalty = (
            frequency_penalty if frequency_penalty is not None else SamplingParams._frequency_penalty
        )
        self.repetition_penalty = (
            repetition_penalty if repetition_penalty is not None else SamplingParams._repetition_penalty
        )
        self.exponential_decay_length_penalty = exponential_decay_length_penalty
        self.temperature = temperature if temperature is not None else SamplingParams._temperature
        self.top_p = top_p if top_p is not None else SamplingParams._top_p
        self.top_k = top_k if top_k is not None else SamplingParams._top_k
        self.ignore_eos = ignore_eos
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.stop_sequences = stop_sequences if stop_sequences is not None else SamplingParams._stop_sequences
        self.skip_special_tokens = skip_special_tokens
        self.add_special_tokens = add_special_tokens
        self.add_spaces_between_special_tokens = add_spaces_between_special_tokens
        self.print_eos_token = print_eos_token
        self.regular_constraint = regular_constraint
        self.allowed_token_ids = allowed_token_ids
        self.group_request_id = group_request_id
        self.move_kv_to_decode_node = move_kv_to_decode_node
        self.suggested_dp_index = suggested_dp_index
        if self.do_sample is False:
            self.temperature = 1.0
            self.top_p = 1.0
            self.top_k = 1
        if (
            self.temperature >= 0.0 and self.temperature < _SAMPLING_EPS
        ):  # temperature is too slow, change to greedy search
            self.temperature = 1.0
            self.top_k = 1
        self.input_penalty = input_penalty
        if self.n is None:
            self.n = self.best_of
        return

    @classmethod
    def load_generation_cfg(cls, weight_dir):
        try:
            generation_cfg = GenerationConfig.from_pretrained(weight_dir, trust_remote_code=True).to_dict()
            cls._do_sample = generation_cfg.get("do_sample", False)
            cls._presence_penalty = generation_cfg.get("presence_penalty", 0.0)
            cls._frequency_penalty = generation_cfg.get("frequency_penalty", 0.0)
            cls._repetition_penalty = generation_cfg.get("repetition_penalty", 1.0)
            cls._temperature = generation_cfg.get("temperature", 1.0)
            cls._top_p = generation_cfg.get("top_p", 1.0)
            cls._top_k = generation_cfg.get("top_k", -1)
            cls._stop_sequences = generation_cfg.get("stop", None)
        except:
            pass

    def verify(self):
        if self.best_of <= 0 or self.best_of > MAX_BEST_OF:
            raise ValueError(f"need 0 < best_of <= {MAX_BEST_OF}, but get {self.best_of}")
        if self.n != self.best_of:
            raise ValueError("current only supported n == best_of")
        if self.n <= 0 or self.n > MAX_BEST_OF or self.n > self.best_of:
            raise ValueError(f"need 0 < n <= {MAX_BEST_OF}, n <= {self.best_of}, but get {self.n}")
        if self.presence_penalty < 0.0:
            raise ValueError(f"presence_penalty must >= 0.0, got {self.presence_penalty}")
        if self.frequency_penalty < 0.0:
            raise ValueError(f"frequency_penalty must >= 0.0, got {self.frequency_penalty}")
        if self.repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must >= 1.0, got {self.repetition_penalty}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must > 0.0, got {self.temperature}")
        if self.top_p <= 0.0 or self.top_p > 1.0:
            raise ValueError(f"top_p must in (0.0, 1.0], got {self.top_p}")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 (disable), or at least 1, got {self.top_k}.")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be at least 1 , got {self.max_new_tokens}.")
        if self.min_new_tokens < 1:
            raise ValueError(f"min_new_tokens must be at least 1 , got {self.min_new_tokens}.")
        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError(
                f"min_new_tokens must <= max_new_tokens, but got min {self.min_new_tokens}, max {self.max_new_tokens}."
            )

        if len(self.exponential_decay_length_penalty) != 2:
            raise ValueError(
                f"exponential_decay_length_penalty must be a tuple of (int, float), \
                got {self.exponential_decay_length_penalty}."
            )
        if (
            not isinstance(self.exponential_decay_length_penalty[0], int)
            or self.exponential_decay_length_penalty[0] < 0
        ):
            raise ValueError(
                f"exponential_decay_length_penalty[0] must be a non-negative integer, \
                got {self.exponential_decay_length_penalty[0]}."
            )
        if (
            not isinstance(self.exponential_decay_length_penalty[1], float)
            or self.exponential_decay_length_penalty[1] < 1.0
        ):
            raise ValueError(
                f"exponential_decay_length_penalty[1] must be a float >= 1.0, \
                got {self.exponential_decay_length_penalty[1]}."
            )
        if self.regular_constraint is not None and not isinstance(self.regular_constraint, str):
            raise ValueError(
                f"regular_expression must be str type, \
                              but get {str(self.regular_constraint)}"
            )

        if self.regular_constraint is not None:
            # check regex format
            try:
                import interegular

                interegular.parse_pattern(self.regular_constraint)
            except Exception as e:
                raise ValueError(f"regular_expression '{self.regular_constraint}' has parse_pattern_error: {str(e)}")

        if not (self.group_request_id is None or isinstance(self.group_request_id, int)):
            raise ValueError(f"group_request_id must be None or int ,but get {self.group_request_id}")

        if not (self.move_kv_to_decode_node is None or isinstance(self.move_kv_to_decode_node, dict)):
            raise ValueError(f"move_kv_to_decode_node must be None or dict, but get {self.move_kv_to_decode_node}")

        if not (self.suggested_dp_index is None or isinstance(self.suggested_dp_index, int)):
            raise ValueError(f"suggested_dp_index must be None or int, but get {self.suggested_dp_index}")

        self._verify_stop_sentences()

        self._verify_allowed_token_ids()

        return

    def _verify_allowed_token_ids(self):
        if self.allowed_token_ids is not None:
            if (not isinstance(self.allowed_token_ids, list)) or (
                not all(isinstance(token_id, int) for token_id in self.allowed_token_ids)
            ):
                raise ValueError(f"allowed_token_ids need format List[int], but get {self.allowed_token_ids}")
            if self.regular_constraint is not None:
                raise ValueError("allowed_token_ids and regular_constraint can not be used in same time")
        return

    def _verify_stop_sentences(self):
        if self.stop_sequences is not None:
            if isinstance(self.stop_sequences, str):
                return
            if isinstance(self.stop_sequences, list):
                all_str = all(isinstance(stop_info, str) for stop_info in self.stop_sequences)
                all_int_list = all(
                    (isinstance(stop_info, list) and isinstance(x, int) for x in stop_info)
                    for stop_info in self.stop_sequences
                )
                if all_str or all_int_list:
                    return
            raise ValueError("stop_sequences only support str, list[str], list[list[int]] type")

    def stop_sentences_to_token_ids(self, tokenizer):
        if self.stop_sequences is None:
            self.stop_sequences = []
        else:
            if isinstance(self.stop_sequences, str):
                self.stop_sequences = [self.stop_sequences]
            new_stop_sequences = []
            for stop_info in self.stop_sequences:
                if isinstance(stop_info, str):
                    stop_str_ids = self._stop_str_to_token_ids(stop_info, tokenizer)
                    if stop_str_ids is not None and len(stop_str_ids) > 0:
                        new_stop_sequences.append(stop_str_ids)
                if isinstance(stop_info, list):
                    if all(isinstance(x, int) for x in stop_info):
                        if len(stop_info) > 0:
                            new_stop_sequences.append(stop_info)
            self.stop_sequences = new_stop_sequences
        return

    def _stop_str_to_token_ids(self, stop_str: str, tokenizer):
        stop_str_ids = tokenizer.encode(stop_str, add_special_tokens=False)
        return stop_str_ids

    def to_dict(self):
        ret = {}
        ret["do_sample"] = self.do_sample
        ret["presence_penalty"] = self.presence_penalty
        ret["frequency_penalty"] = self.frequency_penalty
        ret["repetition_penalty"] = self.repetition_penalty
        ret["exponential_decay_length_penalty"] = self.exponential_decay_length_penalty
        ret["temperature"] = self.temperature
        ret["top_p"] = self.top_p
        ret["top_k"] = self.top_k
        ret["min_new_tokens"] = self.min_new_tokens
        ret["ignore_eos"] = self.ignore_eos
        ret["max_new_tokens"] = self.max_new_tokens
        ret["stop_sequences"] = self.stop_sequences
        ret["best_of"] = self.best_of
        ret["input_penalty"] = self.input_penalty
        ret["regular_constraint"] = self.regular_constraint
        ret["allowed_token_ids"] = self.allowed_token_ids
        ret["move_kv_to_decode_node"] = self.move_kv_to_decode_node
        return ret

    def to_origin_dict(self):
        ret = self.to_dict()
        ret["group_request_id"] = self.group_request_id
        ret["suggested_dp_index"] = self.suggested_dp_index
        return ret

    def to_string(self):
        output_str = ""
        for name, value in vars(self).items():
            output_str += f"{name}: {value} "
        return output_str
