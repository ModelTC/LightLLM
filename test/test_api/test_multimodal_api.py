import base64
import time
import requests
import json
import threading

QUESTION_TEMPLATES = {
    "llava": (
        "<|im_start|>system\n"
        "A chat between a curious human and an artificial intelligence assistant."
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
        "<|im_end|><|im_start|>user\n"
        "<image>\n"
        "Please describe it.\n"
        "<|im_end|><|im_start|>assistant\n"
    ),
    "internvl-internlm2": (
        "<|im_start|>system\n"
        "You are an AI assistant whose name is InternLM(书生·浦语).\n"
        "<|im_end|><|im_start|>user\n"
        "<image>\n"
        "Please describe it.\n"
        "<|im_end|><|im_start|>assistant\n"
    ),
    "internvl-phi3": (
        "<|im_start|>system\n"
        "You are an AI assistant whose name is Phi-3.\n"
        "<|im_end|><|im_start|>user\n"
        "<image>\n"
        "Please describe it.\n"
        "<|im_end|><|im_start|>assistant\n"
    ),
    "internvl2-internlm2": (
        "<|im_start|>system\n"
        "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。\n"
        "<|im_end|><|im_start|>user\n"
        "<image>\n"
        "Please describe it.\n"
        "<|im_end|><|im_start|>assistant\n"
    ),
    "internvl2-phi3": (
        "<|im_start|>system\n"
        "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。\n"
        "<|im_end|><|im_start|>user\n"
        "<image>\n"
        "Please describe it.\n"
        "<|im_end|><|im_start|>assistant\n"
    ),
    "internvl2_5": (
        "<|im_start|>system\n"
        "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。\n"
        "<|im_end|><|im_start|>user\n"
        "<image>\n"
        "Please describe it.\n"
        "<|im_end|><|im_start|>assistant\n"
    ),
    "qwen_vl": (
        "<|im_start|>system\n"
        "You are a helpful assistant.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<img></img>Describe this image.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "qwen2_vl": (
        "<|im_start|>system\n"
        "You are a helpful assistant.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>Describe this image.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "qwen2_5_vl": (
        "<|im_start|>system\n"
        "You are a helpful assistant.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>Describe this image.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
}


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


def image_to_base64(image):
    with open(image, "rb") as fin:
        encoded_string = base64.b64encode(fin.read()).decode("utf-8")
    return encoded_string


# Please replace the question template as QUESTION_TEMPLATES:
question = "Describe this picture to me."
question = (
    f"<|im_start|>system\n"
    f"You are an AI assistant whose name is InternLM(书生·浦语).<|im_end|>"
    f"<|im_start|>user\n<image>\n{question}<|im_end|><|im_start|>assistant\n"
)

url = "http://localhost:9999/generate"
headers = {"Content-Type": "application/json"}

for i in range(1):
    b64 = image_to_base64("test/test.jpg")
    dct = {"type": "base64", "data": b64}
    data = {
        "inputs": question,
        "parameters": {"temperature": 0.0, "do_sample": False, "max_new_tokens": 200},
        "multimodal_params": {
            "images": [
                dct,
            ]
        },
    }
    thread = RequestThread(url, headers, data)
    thread.start()
