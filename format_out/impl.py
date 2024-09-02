import json
import copy
import dataclasses
import requests
from lightllm.server.sampling_params import SamplingParams
from pydantic import BaseModel
from typing import List
from outlines.fsm.json_schema import build_regex_from_schema


from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class ChatSession:

    chat_his: str
    sampling_param: SamplingParams
    url: str = "http://localhost:8017/generate"
    http_headers: dict = dataclasses.field(default_factory=lambda: {"Content-Type": "application/json"})
    default_retry_count: int = 1
    disable_log: bool = False

    def add_prompt(self, data: str):
        self.chat_his += data
        return

    def del_last_prompt(self, len: int):
        self.chat_his = self.chat_his[:-len]
        return

    def generate(self, regex: str = None, max_new_tokens=None, prefix_regex=None, retry_count=1):
        sampling_param = copy.copy(self.sampling_param)
        if max_new_tokens is not None:
            sampling_param.max_new_tokens = max_new_tokens
        if prefix_regex is not None and regex is not None:
            regex = prefix_regex + "(" + regex + ")"

        sampling_param.regular_constraint = regex
        sampling_param.verify()

        data = {"inputs": self.chat_his, "parameters": sampling_param.to_dict()}

        for _ in range(retry_count):
            try:
                response = requests.post(self.url, headers=self.http_headers, data=json.dumps(data))
                if response.status_code == 200:
                    json_ans = response.json()
                    if not self.disable_log:
                        logger.info(f"gen get {str(json_ans)}")
                    return json_ans["generated_text"][0]
                else:
                    logger.warning(f"gen Error: {response.status_code}, {response.text[0:100]}")
                    logger.info("retry gen")
            except:
                pass

        raise Exception("gen error, please check")
        return

    def select(self, args: List[str], max_new_tokens=None, prefix_regex=None):
        if max_new_tokens is None:
            max_new_tokens = max([len(e) for e in args])
        regex = "(" + "|".join(args) + ")"
        return self.generate(
            regex, max_new_tokens=max_new_tokens, prefix_regex=prefix_regex, retry_count=self.default_retry_count
        )

    def gen_int(self, max_new_tokens=None, prefix_regex=None):
        if max_new_tokens is None:
            max_new_tokens = 100
        regex = r"-?\d+"
        return self.generate(
            regex, max_new_tokens=max_new_tokens, prefix_regex=prefix_regex, retry_count=self.default_retry_count
        )

    def gen_float(self, max_new_tokens=None, prefix_regex=None):
        if max_new_tokens is None:
            max_new_tokens = 100
        regex = r"-?\d+\.\d+"
        return self.generate(
            regex, max_new_tokens=max_new_tokens, prefix_regex=prefix_regex, retry_count=self.default_retry_count
        )

    def gen_number(self, max_new_tokens=None, prefix_regex=None):
        if max_new_tokens is None:
            max_new_tokens = 100
        return self.generate(
            r"-?(\d+|\d+\.\d+)",
            max_new_tokens=max_new_tokens,
            prefix_regex=prefix_regex,
            retry_count=self.default_retry_count,
        )

    def gen_number_v2(self, max_new_tokens=None, prefix_regex=None):
        """
        包含分数的支持
        """
        if max_new_tokens is None:
            max_new_tokens = 100
        return self.generate(
            r"-?(\d+(\.\d+)?|\d+/\d+|\d+/\d+\.\d+)",
            max_new_tokens=max_new_tokens,
            prefix_regex=prefix_regex,
            retry_count=self.default_retry_count,
        )

    def gen_json_object(self, obj: BaseModel, max_new_tokens=512, prefix_regex=None, whitespace_pattern=r"[\s]{0,12}"):
        json_schema = obj.model_json_schema()
        regex_str = build_regex_from_schema(json.dumps(json_schema), whitespace_pattern=whitespace_pattern)
        return self.generate(
            regex_str, max_new_tokens=max_new_tokens, prefix_regex=prefix_regex, retry_count=self.default_retry_count
        )
