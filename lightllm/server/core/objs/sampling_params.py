import os
import json
import ctypes
from typing import Optional, List, Tuple, Union
from lightllm.utils.config_utils import get_generation_config_diff_dict
from lightllm.server.req_id_generator import MAX_BEST_OF
from lightllm.utils.envs_utils import get_env_start_args
from .pd_kv_trans_params import PDKVTransParamObj

_SAMPLING_EPS = 1e-5
DEFAULT_INPUT_PENALTY = os.getenv("INPUT_PENALTY", "False").upper() in ["ON", "TRUE", "1"]
SKIP_SPECIAL_TOKENS = os.getenv("SKIP_SPECIAL_TOKENS", "False").upper() in ["ON", "TRUE", "1"]

# 从环境变量获取最大长度限制
STOP_SEQUENCE_MAX_LENGTH = int(os.getenv("LIGHTLLM_STOP_SEQUENCE_MAX_LENGTH", 256))
STOP_SEQUENCE_STR_MAX_LENGTH = int(os.getenv("LIGHTLLM_STOP_SEQUENCE_STR_MAX_LENGTH", 256))
MAX_STOP_SEQUENCES = int(os.getenv("LIGHTLLM_MAX_STOP_SEQUENCES", 10))
REGULAR_CONSTRAINT_MAX_LENGTH = int(os.getenv("LIGHTLLM_REGULAR_CONSTRAINT_MAX_LENGTH", 2048))
# EBNF expanded from a small JSON schema is often several kilobytes.
GRAMMAR_CONSTRAINT_MAX_LENGTH = int(os.getenv("LIGHTLLM_GRAMMAR_CONSTRAINT_MAX_LENGTH", 8192))
JSON_SCHEMA_MAX_LENGTH = int(os.getenv("LIGHTLLM_JSON_SCHEMA_MAX_LENGTH", 2048))
INVALID_TOKEN_IDS_MAX_LENGTH = int(os.getenv("LIGHTLLM_INVALID_TOKEN_IDS_MAX_LENGTH", 10))
MAX_PROMPT_LOGPROBS = int(os.getenv("LIGHTLLM_MAX_PROMPT_LOGPROBS", 1024))
MAX_SEED = (1 << 63) - 1


class StopStrMatchMode(ctypes.Structure):
    """请求级字符串 stop 匹配策略；显式 token ID 序列始终由推理层匹配，不受本模式影响。"""

    _pack_ = 4
    _fields_ = [("mode", ctypes.c_int)]

    # HTTP 输出层对全部生成文本匹配字符串 stop，包括 reasoning 和回答。
    ALL_TEXT = 1
    # HTTP 输出层跳过 reasoning，仅匹配 content。
    CONTENT_ONLY = 2

    def __init__(self, mode: int = ALL_TEXT):
        self.set_mode(mode)

    def set_mode(self, mode: int):
        if mode not in (self.ALL_TEXT, self.CONTENT_ONLY):
            raise ValueError(f"Invalid stop_str_match_mode: {mode!r}")
        self.mode = mode

    def get_mode(self) -> int:
        return self.mode

    def is_all_text(self) -> bool:
        return self.mode == self.ALL_TEXT

    def is_content_only(self) -> bool:
        return self.mode == self.CONTENT_ONLY


class StopSequence(ctypes.Structure):
    """保存一种 stop：字符串仅写 sequence_str，显式 token ID 序列仅写 sequence。"""

    _pack_ = 4
    _fields_ = [
        ("sequence", ctypes.c_int * STOP_SEQUENCE_MAX_LENGTH),
        ("size", ctypes.c_int),
        ("sequence_str", ctypes.c_char * STOP_SEQUENCE_STR_MAX_LENGTH),
        ("sequence_str_len", ctypes.c_int),
    ]

    def initialize(self, sequence: List[int], sequence_str: Optional[str] = None):
        self.size = _check_and_store_int_token_ids(self.sequence, sequence, STOP_SEQUENCE_MAX_LENGTH, "stop token ids")

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
        # 字符串 stop 不支持内嵌 NUL（\x00）；c_char 数组属性读取会在 NUL 处截断。
        return bytes(self.sequence_str[0 : self.sequence_str_len]).decode("utf-8")


class StopSequenceGroups(ctypes.Structure):
    """分别保存字符串和显式 token ID 序列，字符串不经过 tokenizer 编码。"""

    _pack_ = 4
    _fields_ = [
        ("groups", StopSequence * MAX_STOP_SEQUENCES),
        ("size", ctypes.c_int),
    ]

    def initialize(self, stop_sequences: Union[str, List[Union[List[int], str]]]):
        if stop_sequences is None:
            stop_sequences = []
        elif isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]

        stop_sequences = [stop for stop in stop_sequences if stop]
        self.size = len(stop_sequences)
        assert self.size <= MAX_STOP_SEQUENCES, "Too many stop sequence groups."

        for group_idx in range(self.size):
            if isinstance(stop_sequences[group_idx], str):
                self.groups[group_idx].initialize([], sequence_str=stop_sequences[group_idx])
            else:
                assert isinstance(stop_sequences[group_idx], list), "stop sequence must be str or List[int]"
                self.groups[group_idx].initialize(stop_sequences[group_idx])

    def to_token_ids_list(self) -> List[List[int]]:
        """只返回显式 token ID 序列，供推理层在所有阶段匹配。"""
        return [group.to_list() for group in self.groups[: self.size] if group.size > 0]

    def to_origin_list(self) -> List[Union[str, List[int]]]:
        """序列化时保留各 stop 的原始类型和顺序，避免重建请求时丢失字符串。"""
        return [group.to_string() if group.sequence_str_len else group.to_list() for group in self.groups[: self.size]]

    def has_stop_strs(self) -> bool:
        """只检查字符串是否存在，避免仅判断匹配入口时解码和排序。"""
        return any(group.sequence_str_len > 0 for group in self.groups[: self.size])

    def to_stop_str_list(self) -> List[str]:
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

    def initialize(self, constraint: Union[str, dict], tokenizer):
        if isinstance(constraint, dict):
            constraint = json.dumps(constraint, ensure_ascii=False, separators=(",", ":"))
        constraint_bytes = constraint.encode("utf-8")
        if len(constraint_bytes) >= JSON_SCHEMA_MAX_LENGTH:
            raise ValueError("Guided json schema is too long.")

        ctypes.memmove(self.constraint, constraint_bytes, len(constraint_bytes))
        self.length = len(constraint_bytes)
        return

    def to_str(self):
        if self.length == 0:
            return ""
        return bytes(self.constraint[0 : self.length]).decode("utf-8").rstrip("\x00")


class ConstraintOutParams(ctypes.Structure):
    """Static parameters for output constraints and PD continuation."""

    _pack_ = 4
    _fields_ = [
        # 正则表达式约束，限制模型生成的输出文本格式。
        ("regular_constraint", RegularConstraint),
        # EBNF 语法约束；特殊值 "json" 表示输出一个 JSON 对象。
        ("guided_grammar", GuidedGrammar),
        # JSON Schema 约束，限制输出 JSON 的类型、字段及取值。
        ("guided_json", GuidedJsonSchema),
        # 思考结束标记的 token ID 序列；完整匹配后，从下一个 token 开始启用输出约束。
        # 标记本身不参与语法匹配，也不会触发请求停止；为空时立即启用输出约束。
        ("guided_reasoning_end", StopSequence),
        # PD 续跑输入末尾需要重放的 token 数，首段为 0。
        # PD Master 按拼接文本与原始 prompt 的编码长度差设置，推理侧重放该后缀恢复思考和语法进度。
        # 支持从序列化参数恢复该值。
        ("pd_previous_output_len", ctypes.c_int),
    ]

    def initialize(
        self,
        tokenizer,
        regular_constraint: str,
        guided_grammar: str,
        guided_json: Union[str, dict],
        guided_reasoning_end: List[int],
        pd_previous_output_len: int = 0,
    ):
        if not isinstance(pd_previous_output_len, int) or not 0 <= pd_previous_output_len <= 2 ** 31 - 1:
            raise ValueError("pd_previous_output_len must be an integer in [0, 2147483647]")
        self.regular_constraint.initialize(regular_constraint)
        self.guided_grammar.initialize(guided_grammar, tokenizer)
        self.guided_json.initialize(guided_json, tokenizer)
        self.guided_reasoning_end.initialize(guided_reasoning_end)
        self.pd_previous_output_len = pd_previous_output_len

    def has_constraint(self):
        return bool(self.regular_constraint.length or self.guided_grammar.length or self.guided_json.length)


class InvalidTokenIds(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("ids", ctypes.c_int * INVALID_TOKEN_IDS_MAX_LENGTH),
        ("size", ctypes.c_int),
    ]

    def initialize(self, ids: List[int]):
        self.size = _check_and_store_int_token_ids(self.ids, ids, INVALID_TOKEN_IDS_MAX_LENGTH, "invalid token ids")
        return

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


class NodeUUId(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("node_id_high", ctypes.c_uint64),
        ("node_id_low", ctypes.c_uint64),
    ]

    def initialize(self, node_id: int):
        self.node_id_high = (node_id >> 64) & 0xFFFFFFFFFFFFFFFF
        self.node_id_low = node_id & 0xFFFFFFFFFFFFFFFF
        return

    def get(self) -> int:
        return (self.node_id_high << 64) | self.node_id_low


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
        ("min_pixels", ctypes.c_int),
        ("max_pixels", ctypes.c_int),
        ("max_new_tokens", ctypes.c_int),
        ("min_new_tokens", ctypes.c_int),
        # Whether to count input tokens for presence_penalty, frequency_penalty and repetition_penalty
        ("input_penalty", ctypes.c_bool),
        ("constraint_out_params", ConstraintOutParams),
        # if provided, the invalid token ids will be ignored during generation
        ("invalid_token_ids", InvalidTokenIds),
        ("stop_sequences", StopSequenceGroups),
        # 字符串 stop 匹配策略，取值见 StopStrMatchMode；显式 token ID 序列不受本策略影响。
        # 默认 ALL_TEXT；Chat 接口需要跳过 reasoning 时使用 CONTENT_ONLY。
        ("stop_str_match_mode", StopStrMatchMode),
        # 是否保留命中的字符串 stop；False 时移除 stop，True 时保留。
        # 两种设置都会移除同一 token 中 stop 后的文本；不影响 token ID stop 序列。
        ("include_stop_str_in_output", ctypes.c_bool),
        # stop 检测的初始推理状态提示：-1 使用 parser 默认值，0 为回答，1 为思考。
        # 用于初始化当前请求或 PD 续跑段；运行时状态由 StopStrMatchBuffer 维护。
        ("_initial_reasoning_state", ctypes.c_int),
        ("exponential_decay_length_penalty", ExponentialDecayLengthPenalty),
        ("group_request_id", ctypes.c_int64),  # p d mode used params
        # 由 PD Master 为分段续跑或预计 cache 命中率较高的请求设置，表示请求需
        # 以高优先级插入 Router 调度队列。
        ("pd_high_priority_request", ctypes.c_bool),
        # P/D 节点的资源等待超时，由 PD Master 下发。非负值用于控制 shm_req 申请和
        # Router 等待进入推理系统的时限；负数表示永久等待。
        ("pd_node_resource_wait_timeout_seconds", ctypes.c_int),
        ("suggested_dp_index", ctypes.c_int),  # suggest dp index, deepseekv2 dp mode, use to suggest used dp_index
        # in pd split mode, use to keep the id of pd master
        ("pd_master_node_id", NodeUUId),
        # pd params object, only used in pd mode, used to build kv transport connection in prefill and decode
        ("pd_kv_trans_params", PDKVTransParamObj),
        ("skip_special_tokens", ctypes.c_bool),  # whether to skip special tokens when decoding
        ("add_special_tokens", ctypes.c_bool),  # whether to add special tokens when encoding
        (
            "add_spaces_between_special_tokens",
            ctypes.c_bool,
        ),  # whether to add spaces between special tokens when decoding
        ("print_eos_token", ctypes.c_bool),  # eos_id will be always ignored except the value is set to True
        ("disable_prompt_cache", ctypes.c_bool),  # whether to disable prompt cache
        ("seed", ctypes.c_int64),  # random seed
        # -1 disables prompt logprobs; K >= 0 returns only the top-K prompt tokens.
        ("prompt_logprobs", ctypes.c_int),
    ]

    _do_sample: bool = False
    _presence_penalty: float = 0.0
    _frequency_penalty: float = 0.0
    _repetition_penalty: float = 1.0
    _temperature: float = 1.0
    _top_p: float = 1.0
    _top_k: int = -1  # -1 is for all

    def init(self, tokenizer, **kwargs):
        """Initialize and validate sampling parameters synchronously."""
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
        self.min_pixels = kwargs.get("min_pixels", -1)
        self.max_pixels = kwargs.get("max_pixels", -1)
        self.max_new_tokens = kwargs.get("max_new_tokens", 65535)
        self.min_new_tokens = kwargs.get("min_new_tokens", 1)
        self.input_penalty = kwargs.get("input_penalty", DEFAULT_INPUT_PENALTY)
        self.group_request_id = kwargs.get("group_request_id", -1)
        # PD Master 的内部调度信息，不能由外部请求参数开启或修改。
        self.pd_high_priority_request = False
        self.pd_node_resource_wait_timeout_seconds = -1
        self.suggested_dp_index = kwargs.get("suggested_dp_index", -1)

        self.skip_special_tokens = kwargs.get("skip_special_tokens", SKIP_SPECIAL_TOKENS)
        self.disable_prompt_cache = kwargs.get("disable_prompt_cache", False)

        self.add_special_tokens = kwargs.get("add_special_tokens", True)
        self.add_spaces_between_special_tokens = kwargs.get("add_spaces_between_special_tokens", True)
        self.print_eos_token = kwargs.get("print_eos_token", False)
        # ctypes silently wraps overflowing integers assigned to c_int64.
        self.seed = self._normalize_and_verify_seed(kwargs.get("seed"))
        prompt_logprobs = kwargs.get("prompt_logprobs", None)
        self.prompt_logprobs = -1 if prompt_logprobs is None else int(prompt_logprobs)

        self.exponential_decay_length_penalty = ExponentialDecayLengthPenalty()
        self.exponential_decay_length_penalty.initialize(kwargs.get("exponential_decay_length_penalty", (1, 1.0)))

        self.pd_kv_trans_params = PDKVTransParamObj()
        self.pd_kv_trans_params.set(kwargs.get("pd_kv_trans_params", None))
        self.pd_master_node_id = NodeUUId()
        self.pd_master_node_id.initialize(kwargs.get("pd_master_node_id", 0))

        self.constraint_out_params = ConstraintOutParams()
        self.constraint_out_params.initialize(
            tokenizer=tokenizer,
            regular_constraint=kwargs.get("regular_constraint", ""),
            guided_grammar=kwargs.get("guided_grammar", ""),
            guided_json=kwargs.get("guided_json", ""),
            guided_reasoning_end=kwargs.get("guided_reasoning_end", []),
            pd_previous_output_len=kwargs.get("pd_previous_output_len", 0),
        )

        # Initialize stop_sequence_groups
        stop_sequences = kwargs.get("stop_sequences", [])
        self.stop_sequences = StopSequenceGroups()
        self.stop_sequences.initialize(stop_sequences)
        self.stop_str_match_mode.set_mode(kwargs.get("stop_str_match_mode", StopStrMatchMode.ALL_TEXT))
        self.include_stop_str_in_output = kwargs.get("include_stop_str_in_output", False)
        # 运行时状态由 StopStrMatchBuffer 维护，此参数仅用于初始化。
        self._initial_reasoning_state = kwargs.get("_initial_reasoning_state", -1)

        # Initialize invalid_token_ids
        invalid_token_ids = map(int, kwargs.get("logit_bias", {}).keys())
        self.invalid_token_ids = InvalidTokenIds()
        self.invalid_token_ids.initialize(list[int](invalid_token_ids))

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
            generation_cfg = get_generation_config_diff_dict(weight_dir)
            cls._do_sample = generation_cfg.get("do_sample", False)
            cls._presence_penalty = generation_cfg.get("presence_penalty", 0.0)
            cls._frequency_penalty = generation_cfg.get("frequency_penalty", 0.0)
            cls._repetition_penalty = generation_cfg.get("repetition_penalty", 1.0)
            if cls._repetition_penalty is None:
                cls._repetition_penalty = 1.0
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
            raise ValueError(f"max_new_tokens must be at least 1 , got {self.max_new_tokens}.")
        if self.min_new_tokens < 1:
            raise ValueError(f"min_new_tokens must be at least 1 , got {self.min_new_tokens}.")
        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError(
                f"min_new_tokens must <= max_new_tokens, but got min {self.min_new_tokens}, max {self.max_new_tokens}."
            )
        if self.prompt_logprobs < -1 or self.prompt_logprobs > MAX_PROMPT_LOGPROBS:
            raise ValueError(f"prompt_logprobs must be in [-1, {MAX_PROMPT_LOGPROBS}], got {self.prompt_logprobs}")
        if self.prompt_logprobs >= 0 and not get_env_start_args().enable_prompt_logprobs:
            raise ValueError("prompt_logprobs requires --enable_prompt_logprobs")
        self._verify_grammar_constraint()

        return

    @staticmethod
    def _normalize_and_verify_seed(seed: Optional[int]) -> int:
        seed = -1 if seed is None else seed
        if not -1 <= seed <= MAX_SEED:
            raise ValueError(f"seed must be -1 (random), or an integer in [0, {MAX_SEED}], got {seed}")
        return seed

    def _verify_grammar_constraint(self):
        if self.constraint_out_params.guided_grammar.length != 0:
            if self.constraint_out_params.regular_constraint.length != 0:
                raise ValueError("guided_grammar and regular_constraint can not be used in same time")
            if self.constraint_out_params.guided_json.length != 0:
                raise ValueError("guided_grammar and guided_json can not be used in same time")
        return

    async def verify_cpu_async(self, compiler) -> None:
        """Run CPU-heavy validation before admission or HTTP streaming.

        Each validator must offload expensive work to keep the event loop responsive.
        Ordinary sampling parameters are already validated synchronously by init().
        """
        await self.verify_output_constraint(compiler)

    async def verify_output_constraint(self, compiler) -> None:
        """在请求准入或发送流式响应头前完成编译校验；结果由编译器缓存，不保存在采样参数对象上。"""
        await self.get_compiled_grammar(compiler)

    async def get_compiled_grammar(self, compiler) -> Optional[bytes]:
        """获取编译器缓存中的结果，未命中时执行异步编译；无输出约束时返回 None。"""
        for kind, constraint in (
            ("grammar", self.constraint_out_params.guided_grammar),
            ("regex", self.constraint_out_params.regular_constraint),
            ("json", self.constraint_out_params.guided_json),
        ):
            if constraint.length:
                if compiler is None:
                    raise ValueError("Output constraints require --output_constraint_mode xgrammar")
                return await compiler.compile(kind, constraint.to_str())
        return None

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
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "exponential_decay_length_penalty": self.exponential_decay_length_penalty.to_tuple(),
            "stop_sequences": self.stop_sequences.to_origin_list(),
            "stop_str_match_mode": self.stop_str_match_mode.get_mode(),
            "include_stop_str_in_output": self.include_stop_str_in_output,
            "_initial_reasoning_state": self._initial_reasoning_state,
            "best_of": self.best_of,
            "input_penalty": self.input_penalty,
            "regular_constraint": self.constraint_out_params.regular_constraint.to_str(),
            "guided_grammar": self.constraint_out_params.guided_grammar.to_str(),
            "guided_json": self.constraint_out_params.guided_json.to_str(),
            "guided_reasoning_end": self.constraint_out_params.guided_reasoning_end.to_list(),
            "pd_previous_output_len": self.constraint_out_params.pd_previous_output_len,
            "invalid_token_ids": self.invalid_token_ids.to_list(),
            "group_request_id": self.group_request_id,
            "pd_high_priority_request": self.pd_high_priority_request,
            "pd_node_resource_wait_timeout_seconds": self.pd_node_resource_wait_timeout_seconds,
            "skip_special_tokens": self.skip_special_tokens,
            "add_special_tokens": self.add_special_tokens,
            "add_spaces_between_special_tokens": self.add_spaces_between_special_tokens,
            "print_eos_token": self.print_eos_token,
            "disable_prompt_cache": self.disable_prompt_cache,
            "seed": self.seed,
            "prompt_logprobs": self.prompt_logprobs,
        }

    def to_origin_dict(self):
        ret = self.to_dict()
        ret["group_request_id"] = self.group_request_id
        ret["suggested_dp_index"] = self.suggested_dp_index
        return ret


def _check_and_store_int_token_ids(dst, ids: List[int], max_length: int, name: str) -> int:
    """Validate a caller-supplied list of token ids and copy it into a fixed-size ctypes ``c_int`` array.

    The type check runs against the input ``ids`` (not the zero-filled destination buffer), so a non-int
    entry fails fast with a clear message instead of surfacing an opaque ctypes ``TypeError`` at the
    assignment below.

    Args:
        dst: destination ctypes array, declared as ``c_int * max_length``.
        ids: caller-supplied token ids; every element must be an ``int``.
        max_length: capacity of ``dst``.
        name: field name used in the error messages.

    Returns:
        The number of ids written, i.e. ``len(ids)``.
    """
    size = len(ids)
    assert size <= max_length, f"Too many {name}: {size} > {max_length}."
    assert all(isinstance(e, int) for e in ids), f"all {name} must be int."
    dst[:size] = ids[:]
    return size
