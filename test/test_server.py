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
            print('Error:', response.status_code, response.text)


url = 'http://localhost:8000/generate'
headers = {'Content-Type': 'application/json'}

for i in range(1):
    data = {
        'inputs': 'San Francisco is a',
        # 'temperature': 0.1,
        'parameters' : {
            'do_sample': False,
        }
    }
    thread = RequestThread(url, headers, data)
    thread.start()

time.sleep(2)

for i in range(100):
    data = {
        'inputs': 'San Francisco is a',
        'parameters': {
            'do_sample': False,
            'ignore_eos': True,
            'max_new_tokens': 200,
        }
    }
    thread = RequestThread(url, headers, data)
    thread.start()
