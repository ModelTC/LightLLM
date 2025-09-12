import json
import requests
import base64


def run(query, uris):
    images = []
    for uri in uris:
        if uri.startswith("http"):
            images.append({"type": "url", "data": uri})
        else:
            with open(uri, "rb") as fin:
                b64 = base64.b64encode(fin.read()).decode("utf-8")
            images.append({"type": "base64", "data": b64})

    data = {
        "inputs": query,
        "parameters": {
            "max_new_tokens": 200,
            # The space before <|endoftext|> is important,
            # the server will remove the first bos_token_id,
            # but QWen tokenizer does not has bos_token_id
            "stop_sequences": [" <|endoftext|>", " <|im_start|>", " <|im_end|>"],
        },
        "multimodal_params": {
            "images": images,
        },
    }

    url = "http://127.0.0.1:8081/generate"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response


query = """
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<img></img>
这张图片中的文字是什么，告诉我<|im_end|>
<|im_start|>assistant
"""

response = run(uris=["https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"], query=query)

if response.status_code == 200:
    print(f"Result: {response.json()}")
else:
    print(f"Error: {response.status_code}, {response.text}")
