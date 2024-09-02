import os
import sys
import json
from pydantic import BaseModel
from enum import Enum
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from format_out.impl import ChatSession
from format_out.impl import SamplingParams

# server model is Meta-Llama-3-8B-Instruct

system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>you are a smart assistant<|eot_id|>"""
user_start = """<|start_header_id|>user<|end_header_id|>"""
user_end = """<|eot_id|>"""
assistant_start = """<|start_header_id|>assistant<|end_header_id|>"""
assistant_end = """<|eot_id|>"""

chat_session = ChatSession(
    chat_his=system_prompt, url="http://localhost:8017/generate", sampling_param=SamplingParams(do_sample=False)
)
chat_session.sampling_param.top_p = 0.7
chat_session.sampling_param.top_k = 12
chat_session.disable_log = True
# 修改采样参数
chat_session.sampling_param.stop_sequences = [assistant_end, " " + assistant_end, "<|end_of_text|>", " <|end_of_text|>"]

chat_session.add_prompt(user_start)
chat_session.add_prompt("请问1+1+300+2=?")
chat_session.add_prompt(user_end)
chat_session.add_prompt(assistant_start)

print(chat_session.gen_int(max_new_tokens=100))
print(chat_session.generate(max_new_tokens=100))  # 无约束


class Difficulty(Enum):
    Easy = "easy"
    Hard = "hard"
    NORMAL = "normal"


class Result(BaseModel):
    difficulty: Difficulty
    thoughts: List[str]
    answer: str


json_ans = chat_session.gen_json_object(Result, max_new_tokens=300, prefix_regex=r"[\s]{0,20}")
print(json_ans)
formatted_json = json.dumps(json.loads(json_ans), indent=4, ensure_ascii=False)
print(formatted_json)

a = Result(difficulty=Difficulty.Easy, thoughts=["1 + 1 + 300 + 2 = 304"], answer="304")
print(a.model_dump_json(indent=4))

chat_session.add_prompt(formatted_json)
chat_session.add_prompt(assistant_end)
chat_session.add_prompt(user_start)
chat_session.add_prompt("解方程 x + 3x+ 1 =0?")
chat_session.add_prompt(user_end)
chat_session.add_prompt(assistant_start)

json_ans = chat_session.gen_json_object(Result, max_new_tokens=300, prefix_regex=r"[\s]{0,20}")
print(json_ans)
formatted_json = json.dumps(json.loads(json_ans), indent=4, ensure_ascii=False)
print(formatted_json)

json_ans = chat_session.generate(max_new_tokens=300, prefix_regex=r"[\s]{0,20}")
print(json_ans)
