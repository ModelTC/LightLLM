from openai import OpenAI
from datetime import datetime
import argparse
import threading
import random
from typing import List


class OpenAIMultiTurnChat:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", base_url: str = None, client_id: int = 0):
        """
        初始化 OpenAI 多轮对话

        Args:
            api_key: OpenAI API 密钥
            model: 使用的模型名称
            base_url: API 基础 URL（如果使用代理或其他服务）
            client_id: 客户端 ID（用于并发测试）
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.conversation_history = []
        self.client_id = client_id

    def add_message(self, role: str, content: str):
        """添加消息到对话历史"""
        self.conversation_history.append({"role": role, "content": content})

    def get_response(self, user_message: str, verbose: bool = True) -> str:
        """获取 AI 回复（流式）"""
        self.add_message("user", user_message)

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=self.conversation_history, max_tokens=1000, stream=True
            )

            assistant_reply = ""
            if verbose:
                print(f"AI (用户_{self.client_id}): ", end="", flush=True)

            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    assistant_reply += content
                    if verbose:
                        print(content, end="", flush=True)

            if verbose:
                print()  # 换行
            self.add_message("assistant", assistant_reply)

            return assistant_reply

        except Exception as e:
            if verbose:
                print(f"请求失败: {e}")
            return "请求失败，请检查网络连接或 API 密钥"

    def start_conversation(self, system_prompt: str = None):
        """开始新的对话"""
        self.conversation_history = []

        if system_prompt:
            self.add_message("system", system_prompt)

        print(f"开始多轮对话 - 用户_{self.client_id} (输入 'quit' 或 'exit' 退出)")
        print("-" * 50)

        while True:
            user_input = input(f"用户_{self.client_id}: ").strip()

            if user_input.lower() in ["quit", "exit", "退出"]:
                print("对话结束")
                break

            if not user_input:
                continue

            self.get_response(user_input)
            print()


class ParallelChatManager:
    """并发对话管理器"""

    def __init__(self, api_key: str, model: str, base_url: str, parallel: int, system_prompt: str = None):
        """
        初始化并发对话管理器

        Args:
            api_key: OpenAI API 密钥
            model: 使用的模型名称
            base_url: API 基础 URL
            parallel: 并发客户端数量
            system_prompt: 系统提示词
        """
        self.clients: List[OpenAIMultiTurnChat] = []
        self.parallel = parallel
        self.system_prompt = system_prompt

        # 创建多个客户端实例
        for i in range(parallel):
            client = OpenAIMultiTurnChat(api_key=api_key, model=model, base_url=base_url, client_id=i)
            if system_prompt:
                client.add_message("system", system_prompt)
            self.clients.append(client)

    def parallel_request(self, user_message: str):
        """并发发送请求"""
        responses = [None] * self.parallel
        threads = []

        # 随机选择一个客户端来打印输出
        verbose_client_id = random.randint(0, self.parallel - 1)

        def worker(client_idx: int):
            verbose = client_idx == verbose_client_id
            response = self.clients[client_idx].get_response(user_message, verbose=verbose)
            responses[client_idx] = response

        # 启动所有线程
        for i in range(self.parallel):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        return responses

    def start_conversation(self):
        """开始并发对话"""
        print(f"开始并发多轮对话 (并发数: {self.parallel})")
        print("所有客户端输入相同内容，随机显示其中一个客户端的输出")
        print("输入 'quit' 或 'exit' 退出")
        print("-" * 50)

        while True:
            user_input = input("用户输入: ").strip()

            if user_input.lower() in ["quit", "exit", "退出"]:
                print("对话结束")
                break

            if not user_input:
                continue

            print(f"\n[并发请求中... 并发数: {self.parallel}]")
            self.parallel_request(user_input)
            print()


# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI 多轮对话客户端")
    parser.add_argument("--port", type=int, default=13688, help="服务端口号 (默认: 13688)")
    parser.add_argument("--host", type=str, default="localhost", help="服务主机地址 (默认: localhost)")
    parser.add_argument("--api-key", type=str, default="", help="API 密钥 (默认: 空)")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="模型名称 (默认: gpt-3.5-turbo)")
    parser.add_argument("--system-prompt", type=str, default="你是一个有用的助手。", help="系统提示词")
    parser.add_argument("--parallel", type=int, default=1, help="并发客户端数量 (默认: 1, 不并发)")

    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}/v1"

    if args.parallel > 1:
        # 并发模式
        manager = ParallelChatManager(
            api_key=args.api_key,
            model=args.model,
            base_url=base_url,
            parallel=args.parallel,
            system_prompt=args.system_prompt,
        )
        manager.start_conversation()
    else:
        # 单客户端模式
        chat = OpenAIMultiTurnChat(api_key=args.api_key, model=args.model, base_url=base_url)
        chat.start_conversation(args.system_prompt)
