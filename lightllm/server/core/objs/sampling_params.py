import os
import ctypes
from typing import Optional, List, Tuple, Union
from transformers import GenerationConfig
from lightllm.server.req_id_generator import MAX_BEST_OF

_SAMPLING_EPS = 1e-5
DEFAULT_INPUT_PENALTY = os.getenv("INPUT_PENALTY", "False").upper() in ["ON", "TRUE", "1"]
SKIP_SPECIAL_TOKENS = os.getenv("SKIP_SPECIAL_TOKENS", "True").upper() in ["ON", "TRUE", "1"]

# 从环境变量获取最大长度限制
STOP_SEQUENCE_MAX_LENGTH = int(os.getenv("LIGHTLLM_STOP_SEQUENCE_MAX_LENGTH", 256))
STOP_SEQUENCE_STR_MAX_LENGTH = int(os.getenv("LIGHTLLM_STOP_SEQUENCE_STR_MAX_LENGTH", 256))
ALLOWED_TOKEN_IDS_MAX_LENGTH = int(os.getenv("LIGHTLLM_ALLOWED_TOKEN_IDS_MAX_LENGTH", 256))
MAX_STOP_SEQUENCES = int(os.getenv("LIGHTLLM_MAX_STOP_SEQUENCES", 10))
REGULAR_CONSTRAINT_MAX_LENGTH = int(os.getenv("LIGHTLLM_REGULAR_CONSTRAINT_MAX_LENGTH", 2048))
GRAMMAR_CONSTRAINT_MAX_LENGTH = int(os.getenv("LIGHTLLM_GRAMMAR_CONSTRAINT_MAX_LENGTH", 2048))
JSON_SCHEMA_MAX_LENGTH = int(os.getenv("LIGHTLLM_JSON_SCHEMA_MAX_LENGTH", 2048))


class StopSequence(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("sequence", ctypes.c_int * STOP_SEQUENCE_MAX_LENGTH),
        ("size", ctypes.c_int),
        ("sequence_str", ctypes.c_char * STOP_SEQUENCE_STR_MAX_LENGTH),
        ("sequence_str_len", ctypes.c_int),
    ]

    def initialize(self, sequence: List[int], sequence_str: Optional[str] = None):
        self.size = len(sequence)
        assert self.size <= STOP_SEQUENCE_MAX_LENGTH, "stop token length too long."
        assert all(isinstance(e, int) for e in sequence), "all must be int"
        self.sequence[: self.size] = sequence[:]

        if sequence_str is not None:
            sequence_str_bytes = sequence_str.encode("utf-8")
            assert len(sequence_str_bytes) < STOP_SEQUENCE_STR_MAX_LENGTH, "stop sequence string too long."
            self.sequence_str = sequence_str_bytes
            self.sequence_str_len = len(sequence_str_bytes)
        else:
            self.sequence_str_len = 0

    def to_list(self) -> List[int]:
        return list(self.sequence[0 : self.size])

    def to_string(self) -> str:
        return bytes(self.sequence_str[0 : self.sequence_str_len]).decode("utf-8")


class StopSequenceGroups(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("groups", StopSequence * MAX_STOP_SEQUENCES),
        ("size", ctypes.c_int),
    ]

    def initialize(self, stop_sequences: Union[str, List[Union[List[int], str]]], tokenizer):
        if stop_sequences is None:
            stop_sequences = []
        elif isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]

        groups: List[List[int]] = self.stop_sentences_to_token_ids(stop_sequences, tokenizer)
        self.size = len(groups)
        assert self.size <= MAX_STOP_SEQUENCES, "Too many stop sequence groups."

        for group_idx in range(self.size):
            if isinstance(stop_sequences[group_idx], str):
                self.groups[group_idx].initialize(groups[group_idx], sequence_str=stop_sequences[group_idx])
            else:
                self.groups[group_idx].initialize(groups[group_idx])

    def stop_sentences_to_token_ids(self, stop_sequences: List[Union[List[int], str]], tokenizer) -> List[List[int]]:
        new_stop_sequences = []
        for stop_info in stop_sequences:
            if isinstance(stop_info, str):
                stop_str_ids = self._stop_str_to_token_ids(stop_info, tokenizer)
                if stop_str_ids is not None and len(stop_str_ids) > 0:
                    new_stop_sequences.append(stop_str_ids)
            if isinstance(stop_info, list):
                if all(isinstance(x, int) for x in stop_info):
                    if len(stop_info) > 0:
                        new_stop_sequences.append(stop_info)
                else:
                    assert False, "stop_sequences item must be type List[int] when it is a list."
        return new_stop_sequences

    def _stop_str_to_token_ids(self, stop_str: str, tokenizer) -> List[int]:
        stop_str_ids = tokenizer.encode(stop_str, add_special_tokens=False)
        return stop_str_ids

    def to_list(self) -> List[List[int]]:
        return [self.groups[i].to_list() for i in range(self.size)]

    def to_strings(self) -> List[str]:
        # 降序匹配，在出现"\n\n"和"\n"情况时，优先匹配“\n\n”
        return sorted(
            [self.groups[i].to_string() for i in range(self.size) if self.groups[i].sequence_str_len > 0],
            key=len,
            reverse=True,
        )


class RegularConstraint(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("constraint", ctypes.c_ubyte * REGULAR_CONSTRAINT_MAX_LENGTH),
        ("length", ctypes.c_int),
    ]

    def initialize(self, constraint: str):
        constraint_bytes = constraint.encode("utf-8")
        assert len(constraint_bytes) < REGULAR_CONSTRAINT_MAX_LENGTH, "Regular constraint is too long."

        ctypes.memmove(self.constraint, constraint_bytes, len(constraint_bytes))
        self.length = len(constraint_bytes)
        try:
            import interegular

            interegular.parse_pattern(constraint)
        except Exception as e:
            raise ValueError(f"regular_expression '{constraint}' has parse_pattern_error: {str(e)}")
        return

    def to_str(self):
        return bytes(self.constraint[0 : self.length]).decode("utf-8").rstrip("\x00")


class GuidedGrammar(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("constraint", ctypes.c_ubyte * GRAMMAR_CONSTRAINT_MAX_LENGTH),
        ("length", ctypes.c_int),
    ]

    def initialize(self, constraint: str, tokenizer):
        constraint_bytes = constraint.encode("utf-8")
        assert len(constraint_bytes) < GRAMMAR_CONSTRAINT_MAX_LENGTH, "Guided grammar is too long."

        ctypes.memmove(self.constraint, constraint_bytes, len(constraint_bytes))
        self.length = len(constraint_bytes)
        try:
            if self.length > 0 and tokenizer is not None:
                import xgrammar as xgr

                tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
                xgrammar_compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)
                xgrammar_compiler.compile_grammar(constraint)
        except Exception as e:
            raise ValueError(f"guided_grammar '{constraint}' has compile_grammar_error: {str(e)}")
        return

    def to_str(self):
        if self.length == 0:
            return ""
        return bytes(self.constraint[0 : self.length]).decode("utf-8").rstrip("\x00")


class GuidedJsonSchema(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("constraint", ctypes.c_ubyte * JSON_SCHEMA_MAX_LENGTH),
        ("length", ctypes.c_int),
    ]

    def initialize(self, constraint: str, tokenizer):
        constraint_bytes = constraint.encode("utf-8")
        assert len(constraint_bytes) < JSON_SCHEMA_MAX_LENGTH, "Guided json schema is too long."

        ctypes.memmove(self.constraint, constraint_bytes, len(constraint_bytes))
        self.length = len(constraint_bytes)
        try:
            if self.length > 0 and tokenizer is not None:
                import xgrammar as xgr

                tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
                xgrammar_compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)
                xgrammar_compiler.compile_json_schema(constraint)
        except Exception as e:
            raise ValueError(f"guided_grammar '{constraint}' has compile_grammar_error: {str(e)}")
        return

    def to_str(self):
        if self.length == 0:
            return ""
        return bytes(self.constraint[0 : self.length]).decode("utf-8").rstrip("\x00")


class AllowedTokenIds(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("ids", ctypes.c_int * ALLOWED_TOKEN_IDS_MAX_LENGTH),
        ("size", ctypes.c_int),
    ]

    def initialize(self, ids: List[int]):
        self.size = len(ids)
        assert self.size <= ALLOWED_TOKEN_IDS_MAX_LENGTH, "Too many allowed token IDs."
        assert all(isinstance(e, int) for e in self.ids), "all must be int"
        self.ids[: self.size] = ids[:]

    def to_list(self):
        return list(self.ids[: self.size])


class ExponentialDecayLengthPenalty(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("item0", ctypes.c_int),
        ("item1", ctypes.c_float),
    ]

    def initialize(self, inputs: Tuple[int, float]):
        assert len(inputs) == 2, "ExponentialDecayLengthPenalty must be Tuple[int, float]"
        self.item0 = inputs[0]
        assert self.item0 >= 0, "ExponentialDecayLengthPenalty item0 must be int >= 0"
        self.item1 = inputs[1]
        assert self.item1 >= 1.0, "ExponentialDecayLengthPenalty item1 must be a float >= 1.0"
        return

    def to_tuple(self):
        return (self.item0, self.item1)


class DecodeNode(ctypes.Structure):
    _fields_ = [
        ("exists", ctypes.c_bool),
        ("node_id_high", ctypes.c_uint64),  # UUID 的高 64 位
        ("node_id_low", ctypes.c_uint64),  # UUID 的低 64 位
        ("ip", ctypes.c_int32 * 4),
        ("rpyc_port", ctypes.c_int),
        ("max_new_tokens", ctypes.c_int),
        # 记录当前请求使用的 pd_master 节点的 id
        ("pd_master_node_id_high", ctypes.c_uint64),
        ("pd_master_node_id_low", ctypes.c_uint64),
    ]

    def initialize(self, data_dict):
        if data_dict is None:
            self.exists = False
            return

        self.exists = True

        pd_node_id = data_dict["node_id"]
        self.node_id_high = (pd_node_id >> 64) & 0xFFFFFFFFFFFFFFFF
        self.node_id_low = pd_node_id & 0xFFFFFFFFFFFFFFFF

        ip_parts = [int(part) for part in data_dict["ip"].split(".")]
        self.ip = (ctypes.c_int32 * 4)(*ip_parts)

        self.rpyc_port = data_dict["rpyc_port"]
        self.max_new_tokens = data_dict["max_new_tokens"]

        pd_master_node_id = data_dict["pd_master_node_id"]
        self.pd_master_node_id_high = (pd_master_node_id >> 64) & 0xFFFFFFFFFFFFFFFF
        self.pd_master_node_id_low = pd_master_node_id & 0xFFFFFFFFFFFFFFFF

    def to_dict(self):
        if not self.exists:
            return None
        return {
            "node_id": ((self.node_id_high << 64) | self.node_id_low),
            "ip": ".".join(str(self.ip[i]) for i in range(4)),
            "rpyc_port": self.rpyc_port,
            "max_new_tokens": self.max_new_tokens,
            "pd_master_node_id": ((self.pd_master_node_id_high << 64) | self.pd_master_node_id_low),
        }


class SamplingParams(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("best_of", ctypes.c_int),
        ("n", ctypes.c_int),
        ("do_sample", ctypes.c_bool),
        ("presence_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("repetition_penalty", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("top_k", ctypes.c_int),
        ("ignore_eos", ctypes.c_bool),
        # the max number of image patches to be used in the internvl model, for the test
        ("image_max_patch_num", ctypes.c_int),
        ("max_new_tokens", ctypes.c_int),
        ("min_new_tokens", ctypes.c_int),
        # Whether to count input tokens for presence_penalty, frequency_penalty and repetition_penalty
        ("input_penalty", ctypes.c_bool),
        ("regular_constraint", RegularConstraint),
        ("guided_grammar", GuidedGrammar),
        ("guided_json", GuidedJsonSchema),
        # If provided, the engine will construct a logits,
        # processor which only retains scores for the given token ids. Defaults to None.
        # allowed_token_ids only can be used in "--output_constraint_mode outlines" started server.
        ("allowed_token_ids", AllowedTokenIds),
        ("stop_sequences", StopSequenceGroups),
        ("exponential_decay_length_penalty", ExponentialDecayLengthPenalty),
        ("group_request_id", ctypes.c_int64),  # p d mode used params
        ("suggested_dp_index", ctypes.c_int),  # suggest dp index, deepseekv2 dp mode, use to suggest used dp_index
        ("move_kv_to_decode_node", DecodeNode),  # move kv to deocde node, only used in pd mode
        ("skip_special_tokens", ctypes.c_bool),  # whether to skip special tokens when decoding
        ("add_special_tokens", ctypes.c_bool),  # whether to add special tokens when encoding
        (
            "add_spaces_between_special_tokens",
            ctypes.c_bool,
        ),  # whether to add spaces between special tokens when decoding
        ("print_eos_token", ctypes.c_bool),  # eos_id will be always ignored except the value is set to True
    ]

    _do_sample: bool = False
    _presence_penalty: float = 0.0
    _frequency_penalty: float = 0.0
    _repetition_penalty: float = 1.0
    _temperature: float = 1.0
    _top_p: float = 1.0
    _top_k: int = -1  # -1 is for all

    def init(self, tokenizer, **kwargs):
        super().__init__()
        self.best_of = kwargs.get("best_of", 1)
        self.n = kwargs.get("n", self.best_of)
        self.do_sample = kwargs.get("do_sample", SamplingParams._do_sample)
        self.presence_penalty = kwargs.get("presence_penalty", SamplingParams._presence_penalty)
        self.frequency_penalty = kwargs.get("frequency_penalty", SamplingParams._frequency_penalty)
        self.repetition_penalty = kwargs.get("repetition_penalty", SamplingParams._repetition_penalty)
        self.temperature = kwargs.get("temperature", SamplingParams._temperature)
        self.top_p = kwargs.get("top_p", SamplingParams._top_p)
        self.top_k = kwargs.get("top_k", SamplingParams._top_k)
        self.ignore_eos = kwargs.get("ignore_eos", False)
        self.image_max_patch_num = kwargs.get("image_max_patch_num", -1)
        self.max_new_tokens = kwargs.get("max_new_tokens", 16)
        self.min_new_tokens = kwargs.get("min_new_tokens", 1)
        self.input_penalty = kwargs.get("input_penalty", DEFAULT_INPUT_PENALTY)
        self.group_request_id = kwargs.get("group_request_id", -1)
        self.suggested_dp_index = kwargs.get("suggested_dp_index", -1)

        self.skip_special_tokens = kwargs.get("skip_special_tokens", SKIP_SPECIAL_TOKENS)

        self.add_special_tokens = kwargs.get("add_special_tokens", True)
        self.add_spaces_between_special_tokens = kwargs.get("add_spaces_between_special_tokens", True)
        self.print_eos_token = kwargs.get("print_eos_token", False)

        self.exponential_decay_length_penalty = ExponentialDecayLengthPenalty()
        self.exponential_decay_length_penalty.initialize(kwargs.get("exponential_decay_length_penalty", (1, 1.0)))

        self.move_kv_to_decode_node = DecodeNode()
        self.move_kv_to_decode_node.initialize(kwargs.get("move_kv_to_decode_node", None))

        # Initialize regular_constraint
        regular_constraint = kwargs.get("regular_constraint", "")
        self.regular_constraint = RegularConstraint()
        self.regular_constraint.initialize(regular_constraint)

        # Initialize guided_grammar
        guided_grammar = kwargs.get("guided_grammar", "")
        self.guided_grammar = GuidedGrammar()
        self.guided_grammar.initialize(guided_grammar, tokenizer)

        # Initialize guided_json
        guided_json = kwargs.get("guided_json", "")
        self.guided_json = GuidedJsonSchema()
        self.guided_json.initialize(guided_json, tokenizer)

        # Initialize stop_sequence_groups
        stop_sequences = kwargs.get("stop_sequences", [])
        self.stop_sequences = StopSequenceGroups()
        self.stop_sequences.initialize(stop_sequences, tokenizer)

        # Initialize allowed_token_ids
        allowed_token_ids = kwargs.get("allowed_token_ids", [])
        self.allowed_token_ids = AllowedTokenIds()
        self.allowed_token_ids.initialize(allowed_token_ids)

        if self.do_sample is False:
            self.temperature = 1.0
            self.top_p = 1.0
            self.top_k = 1
        if (
            self.temperature >= 0.0 and self.temperature < _SAMPLING_EPS
        ):  # temperature is too slow, change to greedy search
            self.temperature = 1.0
            self.top_k = 1

        self.verify()

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
        except:
            pass

    def verify(self):
        if self.best_of <= 0 or self.best_of > MAX_BEST_OF:
            raise ValueError(f"need 0 < best_of <= {MAX_BEST_OF}, but got {self.best_of}")
        if self.n != self.best_of:
            raise ValueError("current only supported n == best_of")
        if self.n <= 0 or self.n > MAX_BEST_OF or self.n > self.best_of:
            raise ValueError(f"need 0 < n <= {MAX_BEST_OF}, n <= {self.best_of}, but got {self.n}")
        if self.presence_penalty < 0.0:
            raise ValueError(f"presence_penalty must >= 0.0, got {self.presence_penalty}")
        if self.frequency_penalty < 0.0:
            raise ValueError(f"frequency_penalty must >= 0.0, got {self.frequency_penalty}")
        if self.repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must >= 1.0, got {self.repetition_penalty}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must > 0.0, got {self.temperature}")
        if self.top_p <= 0.0 or self.top_p > 1.0:
            raise ValueError(f"top_p must be in (0.0, 1.0], got {self.top_p}")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 (disable), or at least 1, got {self.top_k}.")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be at least 1, got {self.max_new_tokens}.")
        if self.min_new_tokens < 1:
            raise ValueError(f"min_new_tokens must be at least 1, got {self.min_new_tokens}.")
        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError(
                f"min_new_tokens must <= max_new_tokens, but got min {self.min_new_tokens}, max {self.max_new_tokens}."
            )

        self._verify_allowed_token_ids()
        self._verify_grammar_constraint()

        return

    def _verify_grammar_constraint(self):
        if self.guided_grammar.length != 0:
            if self.regular_constraint.length != 0:
                raise ValueError("guided_grammar and regular_constraint can not be used in same time")
            if self.guided_json.length != 0:
                raise ValueError("guided_grammar and guided_json can not be used in same time")
        return

    def _verify_allowed_token_ids(self):
        if self.allowed_token_ids.size != 0:
            if self.regular_constraint.length != 0:
                raise ValueError("allowed_token_ids and regular_constraint can not be used in same time")
            if self.guided_grammar.length != 0:
                raise ValueError("allowed_token_ids and guided_grammar can not be used in same time")
            if self.guided_json.length != 0:
                raise ValueError("allowed_token_ids and guided_json can not be used in same time")
        return

    def to_dict(self):
        return {
            "do_sample": self.do_sample,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "ignore_eos": self.ignore_eos,
            "image_max_patch_num": self.image_max_patch_num,
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "exponential_decay_length_penalty": self.exponential_decay_length_penalty.to_tuple(),
            "stop_sequences": self.stop_sequences.to_list(),
            "best_of": self.best_of,
            "input_penalty": self.input_penalty,
            "regular_constraint": self.regular_constraint.to_str(),
            "guided_grammar": self.guided_grammar.to_str(),
            "guided_json": self.guided_json.to_str(),
            "allowed_token_ids": self.allowed_token_ids.to_list(),
            "group_request_id": self.group_request_id,
            "move_kv_to_decode_node": self.move_kv_to_decode_node.to_dict(),
            "skip_special_tokens": self.skip_special_tokens,
            "add_special_tokens": self.add_special_tokens,
            "add_spaces_between_special_tokens": self.add_spaces_between_special_tokens,
            "print_eos_token": self.print_eos_token,
        }

    def to_origin_dict(self):
        ret = self.to_dict()
        ret["group_request_id"] = self.group_request_id
        ret["suggested_dp_index"] = self.suggested_dp_index
        return ret
