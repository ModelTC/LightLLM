import os


def read_md_files(root_dir):
    # 遍历目录
    ans_str = ""
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".md"):
                file_path = os.path.join(dirpath, filename)
                # 读取文件内容
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    print(f"Path: {file_path}\nContent:\n{content}\n")
                    ans_str += f"<title>{file_path}</title><content>{content}</content>\n\n"
    return ans_str


pdf_str = read_md_files("./")
import sys
import json
from pydantic import BaseModel, constr, conlist
from enum import Enum
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from format_out.impl import ChatSession
from format_out.impl import SamplingParams

# server model is Meta-Llama-3-8B-Instruct
system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a smart assistant<|eot_id|>"""
user_start = """<|start_header_id|>user<|end_header_id|>"""
user_end = """<|eot_id|>"""
assistant_start = """<|start_header_id|>assistant<|end_header_id|>"""
assistant_end = """<|eot_id|>"""
knowledge_start = """<|start_header_id|>knowledge<|end_header_id|>"""
knowledge_end = """<|eot_id|>"""

chat_session = ChatSession(
    chat_his=system_prompt, url="http://localhost:8017/generate", sampling_param=SamplingParams(do_sample=False)
)
chat_session.sampling_param.top_p = 0.7
chat_session.sampling_param.top_k = 12
chat_session.disable_log = True
# 修改采样参数
chat_session.sampling_param.stop_sequences = [assistant_end, " " + assistant_end, "<|end_of_text|>", " <|end_of_text|>"]

chat_session.add_prompt(knowledge_start)
chat_session.add_prompt(pdf_str)
chat_session.add_prompt(knowledge_end)

chat_session.add_prompt(user_start)
chat_session.add_prompt("你觉得知识库中的内容是与什么领域相关的")
chat_session.add_prompt(user_end)

chat_session.add_prompt(assistant_start)


class Result(BaseModel):
    finded_relevant_knowledge: conlist(str, min_length=1, max_length=10)
    can_answer: bool
    result: constr(min_length=3, max_length=200)


json_ans_str = chat_session.gen_json_object(Result, max_new_tokens=1000, prefix_regex=r"[\s]{0,20}")
print("tmp:", json_ans_str)
json_ans_str: str = json_ans_str.strip()
json_ans_str = json_ans_str.replace("”", '"')  # 修复 json 格式问题
json_ans = json.loads(json_ans_str)
formatted_json = json.dumps(json_ans, indent=4, ensure_ascii=False)
print("assistant:")
print(formatted_json)
chat_session.add_prompt(formatted_json)
chat_session.add_prompt(assistant_end)

chat_session.add_prompt(user_start)
chat_session.add_prompt("从知识库中查找一下llama13b的相关性能数据")
chat_session.add_prompt(user_end)
chat_session.add_prompt(assistant_start)

json_ans_str = chat_session.gen_json_object(Result, max_new_tokens=1000, prefix_regex=r"[\s]{0,20}")
print("tmp:", json_ans_str)
json_ans_str: str = json_ans_str.strip()
json_ans_str = json_ans_str.replace("”", '"')  # 修复 json 格式问题
json_ans = json.loads(json_ans_str)
formatted_json = json.dumps(json_ans, indent=4, ensure_ascii=False)


chat_session.add_prompt(user_start)
chat_session.add_prompt("关于L40s的信息，尽量详细")
chat_session.add_prompt(user_end)
chat_session.add_prompt(assistant_start)

json_ans_str = chat_session.gen_json_object(Result, max_new_tokens=1000, prefix_regex=r"[\s]{0,20}")
print("tmp:", json_ans_str)
json_ans_str: str = json_ans_str.strip()
json_ans_str = json_ans_str.replace("”", '"')  # 修复 json 格式问题
json_ans = json.loads(json_ans_str)
formatted_json = json.dumps(json_ans, indent=4, ensure_ascii=False)
