from openai import OpenAI
from datetime import datetime
import argparse
import threading
import random
import time
import sys
from typing import List

# 尝试导入 readline 以支持更好的中文输入处理
try:
    import readline
except ImportError:
    # Windows 上可能没有 readline，尝试使用 pyreadline3
    try:
        import pyreadline3 as readline
    except ImportError:
        readline = None


def safe_input(prompt: str) -> str:
    """
    安全的输入函数，处理中文输入删除乱码问题
    """
    if readline is not None:
        # readline 已加载，直接使用 input
        return input(prompt)
    else:
        # 没有 readline，使用替代方案
        sys.stdout.write(prompt)
        sys.stdout.flush()
        return sys.stdin.readline().rstrip("\n")


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

    def get_response(self, user_message: str, verbose: bool = True) -> dict:
        """
        获取 AI 回复（流式）

        Returns:
            dict: 包含 response, ttft_ms, avg_itl_ms, total_tokens, total_time_ms
        """
        self.add_message("user", user_message)

        try:
            start_time = time.perf_counter()
            response = self.client.chat.completions.create(
                model=self.model, messages=self.conversation_history, max_tokens=1000, stream=True
            )

            assistant_reply = ""
            first_token_time = None
            token_times = []
            token_count = 0

            if verbose:
                print(f"AI (用户_{self.client_id}): ", end="", flush=True)

            for chunk in response:
                current_time = time.perf_counter()

                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content

                    # 记录第一个 token 的时间 (TTFT)
                    if first_token_time is None:
                        first_token_time = current_time

                    # 记录每个 token 的时间用于计算 ITL
                    token_times.append(current_time)
                    token_count += 1

                    assistant_reply += content
                    if verbose:
                        print(content, end="", flush=True)

            end_time = time.perf_counter()

            # 计算指标
            ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
            total_time_ms = (end_time - start_time) * 1000

            # 计算平均 ITL (Inter-Token Latency)
            avg_itl_ms = 0
            if len(token_times) > 1:
                inter_token_latencies = []
                for i in range(1, len(token_times)):
                    itl = (token_times[i] - token_times[i - 1]) * 1000
                    inter_token_latencies.append(itl)
                avg_itl_ms = sum(inter_token_latencies) / len(inter_token_latencies)

            if verbose:
                print()  # 换行
                # 显示性能指标
                print(
                    f"[TTFT: {ttft_ms:.2f}ms | Avg ITL: {avg_itl_ms:.2f}ms | "
                    f"Tokens: {token_count} | Total: {total_time_ms:.2f}ms]"
                )

            self.add_message("assistant", assistant_reply)

            return {
                "response": assistant_reply,
                "ttft_ms": ttft_ms,
                "avg_itl_ms": avg_itl_ms,
                "total_tokens": token_count,
                "total_time_ms": total_time_ms,
            }

        except Exception as e:
            if verbose:
                print(f"请求失败: {e}")
            return {
                "response": "请求失败，请检查网络连接或 API 密钥",
                "ttft_ms": 0,
                "avg_itl_ms": 0,
                "total_tokens": 0,
                "total_time_ms": 0,
            }

    def start_conversation(self, system_prompt: str = None):
        """开始新的对话"""
        self.conversation_history = []

        if system_prompt:
            self.add_message("system", system_prompt)

        print(f"开始多轮对话 - 用户_{self.client_id} (输入 'quit' 或 'exit' 退出)")
        print("-" * 50)

        while True:
            try:
                user_input = safe_input(f"用户_{self.client_id}: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n对话结束")
                break

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

    def parallel_request(self, user_message: str) -> List[dict]:
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
            try:
                user_input = safe_input("用户输入: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n对话结束")
                break

            if user_input.lower() in ["quit", "exit", "退出"]:
                print("对话结束")
                break

            if not user_input:
                continue

            print(f"\n[并发请求中... 并发数: {self.parallel}]")
            responses = self.parallel_request(user_input)

            # 显示聚合的性能统计
            valid_responses = [r for r in responses if r and r.get("ttft_ms", 0) > 0]
            if valid_responses:
                avg_ttft = sum(r["ttft_ms"] for r in valid_responses) / len(valid_responses)
                avg_itl = sum(r["avg_itl_ms"] for r in valid_responses) / len(valid_responses)
                total_tokens = sum(r["total_tokens"] for r in valid_responses)
                avg_total_time = sum(r["total_time_ms"] for r in valid_responses) / len(valid_responses)
                print(
                    f"\n[聚合统计 - Avg TTFT: {avg_ttft:.2f}ms | Avg ITL: {avg_itl:.2f}ms | "
                    f"Total Tokens: {total_tokens} | Avg Time: {avg_total_time:.2f}ms]"
                )
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
