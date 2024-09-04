import os
import sys
import json
from pydantic import BaseModel, constr, conlist
from enum import Enum
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from format_out.impl import ChatSession
from format_out.impl import SamplingParams

# server model is Meta-Llama-3-8B-Instruct
system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                 你是一个人工智能助手，只会做以下的事情:
                 1. 知识问答 (根据已有的知识库信息,回答对应的问题）
                 2. 查询占用端口的进程号 (通过生成指令，然后由系统执行返回结果，你做总结)
                 3. 其他类型 (所有不是上面两种类型的问题)
                 你在回答相关问题的过程中，需要按照指引，生成相关的json格式输出。
                <|eot_id|>"""
user_start = """<|start_header_id|>user<|end_header_id|>"""
user_end = """<|eot_id|>"""
assistant_start = """<|start_header_id|>assistant<|end_header_id|>"""
assistant_end = """<|eot_id|>"""
knowledge_start = """<|start_header_id|>knowledge<|end_header_id|>"""
knowledge_end = """<|eot_id|>"""


class QaBot:
    def __init__(self, llm_url="http://localhost:8017/generate"):
        chat_session = ChatSession(chat_his=system_prompt, url=llm_url, sampling_param=SamplingParams(do_sample=False))
        chat_session.sampling_param.top_p = 0.7
        chat_session.sampling_param.top_k = 12
        chat_session.disable_log = True
        # 修改采样参数
        chat_session.sampling_param.stop_sequences = [assistant_end, "<|end_of_text|>"]

        # 添加知识库
        chat_session.add_prompt(knowledge_start)

        title = "lightllm的仓库链接"
        content = "https://github.com/ModelTC/lightllm"
        chat_session.add_prompt(f"<item><title>{title}</title><content>{content}</content></item>\n")
        title = "lightllm的文档链接"
        content = "https://github.com/ModelTC/lightllm/tree/main/docs"
        chat_session.add_prompt(f"<item><title>{title}</title><content>{content}</content></item>\n")
        title = "steam 账号"
        content = "account:12312321312, password:xxxxxxx"
        chat_session.add_prompt(f"<item><title>{title}</title><content>{content}</content></item>\n")
        title = "今天的天气"
        content = "天气很热"
        chat_session.add_prompt(f"<item><title>{title}</title><content>{content}</content></item>\n")

        chat_session.add_prompt(knowledge_end)

        self.chat_session = chat_session

    def answer(self, question_str: str):
        self.chat_session.add_prompt(user_start)
        self.chat_session.add_prompt(question_str)
        self.chat_session.add_prompt(user_end)

        self.chat_session.add_prompt(assistant_start)
        self.chat_session.add_prompt("先判断该问题的类型:")

        class QAType(Enum):
            Q1 = "知识问答"
            Q2 = "查询占用端口的进程号"
            Q3 = "其他类型"

        class ClassQuestion(BaseModel):
            thoughts: List[str]
            question_type: QAType

        json_ans_str = self.chat_session.gen_json_object(ClassQuestion, max_new_tokens=2048, prefix_regex=r"[\s]{0,20}")
        json_ans_str: str = json_ans_str.strip()
        json_ans_str = json_ans_str.replace("”", '"')  # 修复 json 格式问题
        print(json_ans_str)
        # 修复中文格式问题
        json_ans_str = json_ans_str.encode("unicode_escape").decode()
        json_ans_str = json_ans_str.replace(r"\\u", r"\u")
        json_ans_str = json_ans_str.replace(r"\\\u", r"\u")
        json_ans_str = json_ans_str.replace(r"\\\\u", r"\u")
        json_ans_str = json_ans_str.encode("utf-8").decode("unicode_escape")
        print(json_ans_str)
        json_ans = json.loads(json_ans_str)
        formatted_json = json.dumps(json_ans, indent=4, ensure_ascii=False)
        print(formatted_json)
        self.chat_session.add_prompt(formatted_json + "\n")

        class_ans = ClassQuestion(**json_ans)
        if class_ans.question_type == QAType.Q3:
            ans_str = "对不起,我无法处理这个问题, 我只会下列问题:" "1. 知识问答 (根据已有的知识库信息,回答对应的问题）" "2. 查询占用端口的进程号 (通过生成指令，然后由系统执行返回结)"
            self.chat_session.add_prompt("给用户回答:" + ans_str)
            self.chat_session.add_prompt(assistant_end)
            return ans_str

        elif class_ans.question_type == QAType.Q1:
            return self.handle_qa()
        elif class_ans.question_type == QAType.Q2:
            return self.query_port_pid()

    def handle_qa(self):
        self.chat_session.add_prompt("通过知识库来会的这个问题:")

        class Result(BaseModel):
            finded_relevant_knowledge: conlist(str, min_length=0, max_length=10)
            can_answer: bool
            preliminary_results: str
            summary_result: constr(min_length=0, max_length=1000)

        json_ans_str = self.chat_session.gen_json_object(Result, max_new_tokens=2048, prefix_regex=r"[\s]{0,20}")
        json_ans_str: str = json_ans_str.strip()
        json_ans_str = json_ans_str.replace("”", '"')  # 修复 json 格式问题

        json_ans_str = json_ans_str.encode("unicode_escape").decode()
        json_ans_str = json_ans_str.replace(r"\\u", r"\u")
        json_ans_str = json_ans_str.replace(r"\\\u", r"\u")
        json_ans_str = json_ans_str.replace(r"\\\\u", r"\u")
        json_ans_str = json_ans_str.encode("utf-8").decode("unicode_escape")

        json_ans = json.loads(json_ans_str)
        formatted_json = json.dumps(json_ans, indent=4, ensure_ascii=False)
        print(formatted_json)
        self.chat_session.add_prompt(formatted_json + "\n")
        result = Result(**json_ans)
        if result.can_answer is False:
            ans_str = "对不起,我无法处理这个"
            self.chat_session.add_prompt("给用户回答:" + ans_str)
            self.chat_session.add_prompt(assistant_end)
            return ans_str
        else:
            ans_str = result.summary_result
            self.chat_session.add_prompt("给用户回答:" + ans_str)
            self.chat_session.add_prompt(assistant_end)
            return ans_str

    def query_port_pid(self):
        self.chat_session.add_prompt("收集需要用到的命令信息，如端口号:")

        class Result(BaseModel):
            thoughts: List[str]
            port_can_be_determined: bool
            port: str

        json_ans_str = self.chat_session.gen_json_object(Result, max_new_tokens=2048, prefix_regex=r"[\s]{0,20}")
        json_ans_str: str = json_ans_str.strip()
        json_ans_str = json_ans_str.replace("”", '"')  # 修复 json 格式问题

        json_ans_str = json_ans_str.encode("unicode_escape").decode()
        json_ans_str = json_ans_str.replace(r"\\u", r"\u")
        json_ans_str = json_ans_str.replace(r"\\\u", r"\u")
        json_ans_str = json_ans_str.replace(r"\\\\u", r"\u")
        json_ans_str = json_ans_str.encode("utf-8").decode("unicode_escape")

        json_ans = json.loads(json_ans_str)
        formatted_json = json.dumps(json_ans, indent=4, ensure_ascii=False)
        print(formatted_json)
        self.chat_session.add_prompt(formatted_json + "\n")
        result = Result(**json_ans)
        if result.port_can_be_determined is False:
            ans_str = "对不起,我无法确认端口号，请给出明确的端口号信息"
            self.chat_session.add_prompt("给用户回答:" + ans_str)
            self.chat_session.add_prompt(assistant_end)
            return ans_str
        else:
            import subprocess

            command = f"netstat -tunlp | grep {result.port}"
            result = subprocess.run(command, capture_output=True, text=True, shell=True)

            ans_str = str(result.stdout) + "\n" + str(result.stderr)
            self.chat_session.add_prompt("给用户回答:" + ans_str)
            self.chat_session.add_prompt(assistant_end)
            return ans_str
