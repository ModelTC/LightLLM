import time
import requests
import json
import threading
from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("/data/nvme0/models/llama3-8b-instruct")

ds = load_dataset("NousResearch/json-mode-eval")
prompt = ds["train"]["prompt"]


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


url = "http://localhost:8888/generate"
headers = {"Content-Type": "application/json"}

messages = []
for i in range(80):
    messages.append([{"role": "user", "content": prompt[i]}])

# inputs = tokenizer.apply_chat_template(cot_question, tokenize=False)
inputs = [tokenizer.apply_chat_template(messages[i], tokenize=False) for i in range(len(messages))]
for i in range(10):
    data = {
        "inputs": inputs[i],
        "parameters": {
            "do_sample": False,
            "max_new_tokens": 150,
        },
    }
    thread = RequestThread(url, headers, data)
    thread.start()
