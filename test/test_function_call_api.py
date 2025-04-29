import time
import requests
import json
import threading


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


openai_url = "http://localhost:8888/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# Test OpenAI Tool Call API
messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston today? "
        "Output a reasoning before act, then use the tools to help you.",
    }
]
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for, e.g. 'San Francisco'",
                    },
                    "state": {
                        "type": "string",
                        "description": "the two-letter abbreviation for the state that the city is"
                        " in, e.g. 'CA' which would mean 'California'",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "state", "unit"],
            },
        },
    }
]

for i in range(1):
    data = {
        "model": "qwen25",
        "messages": messages,
        "tools": tools,
        "do_sample": False,
        "max_tokens": 1024,
    }
    thread = RequestThread(openai_url, headers, data)
    thread.start()
