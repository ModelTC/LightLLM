import time
import requests
import json
import threading

"""
python -m lightllm.server.api_server --model_dir /Meta-Llama-3-8B-Instruct  \
                                     --host 0.0.0.0                 \
                                     --port 8017                   \
                                     --tp 1                         \
                                     --max_total_token_num 100000 \
                                     --simple_constraint_mode \
                                     --use_dynamic_prompt_cache
"""


class RequestThread(threading.Thread):
    def __init__(self, url, headers, data):
        threading.Thread.__init__(self)
        self.url = url
        self.headers = headers
        self.data = data

    def run(self):
        response = requests.post(self.url, headers=self.headers, data=json.dumps(self.data))
        if response.status_code == 200:
            print(response.json())
        else:
            print("Error:", response.status_code, response.text)


url = "http://localhost:8017/generate"
headers = {"Content-Type": "application/json"}

for i in range(1):
    data = {
        "inputs": "(100+1+3)*2=",
        # 'temperature': 0.1,
        "parameters": {"do_sample": False, "regular_constraint": r"-?\d+"},
    }
    thread = RequestThread(url, headers, data)
    thread.start()

time.sleep(2)

for i in range(20):
    data = {
        "inputs": "Are dog a man? ",
        "parameters": {
            "do_sample": False,
            "ignore_eos": True,
            "max_new_tokens": 200,
            "regular_constraint": r"(Yes|No) Reason is [a-zA-Z\s]+",
        },
    }
    thread = RequestThread(url, headers, data)
    thread.start()
