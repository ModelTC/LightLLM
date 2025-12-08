.. _openai_api:

LightLLM OpenAI API Usage Examples
==================================

LightLLM provides an interface that is fully compatible with OpenAI API, supporting all standard OpenAI features including function calling. This document provides detailed information on how to use LightLLM's OpenAI interface.

Basic Configuration
-------------------

First, ensure that the LightLLM service is started:

.. code-block:: bash

    # Start LightLLM service
    python -m lightllm.server.api_server \
        --model_dir /path/to/your/model \
        --port 8088 \
        --tp 1

Basic Conversation Examples
---------------------------

1. Simple Conversation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import requests
    import json

    # Configuration
    url = "http://localhost:8088/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    # Request data
    data = {
        "model": "your_model_name",
        "messages": [
            {"role": "user", "content": "Hello, please introduce yourself"}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }

    # Send request
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print("Reply:", result["choices"][0]["message"]["content"])
    else:
        print("Error:", response.status_code, response.text)

2. Streaming Conversation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import requests
    import json

    url = "http://localhost:8088/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    data = {
        "model": "your_model_name",
        "messages": [
            {"role": "user", "content": "Please write a short essay about artificial intelligence"}
        ],
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 1000
    }

    # Streaming request
    response = requests.post(url, headers=headers, json=data, stream=True)
    
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data_str)
                        if chunk['choices'][0]['delta'].get('content'):
                            print(chunk['choices'][0]['delta']['content'], end='', flush=True)
                    except json.JSONDecodeError:
                        continue
    else:
        print("Error:", response.status_code, response.text)

Function Calling Examples
-------------------------

LightLLM supports OpenAI's function calling functionality, providing function call parsing for three models. Specify the --tool_call_parser parameter when starting the service to choose. The service launch command is:

.. code-block:: bash

    python -m lightllm.server.api_server \
        --model_dir /path/to/your/model \
        --port 8088 \
        --tp 1 \
        --tool_call_parser qwen25
    # Optional parameters are qwen25, llama3, mistral

1. Basic Function Calling
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import requests
    import json

    url = "http://localhost:8088/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    # Define functions
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get current weather information for a specified city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name, e.g.: Beijing, Shanghai"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    # Request data
    data = {
        "model": "your_model_name",
        "messages": [
            {"role": "user", "content": "What's the weather like in Beijing today?"}
        ],
        "tools": tools,
        "tool_choice": "auto",  # Let the model automatically decide whether to call functions
        "temperature": 0.7,
        "max_tokens": 1000
    }

    # Send request
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        message = result["choices"][0]["message"]
        
        # Check if there are function calls
        if message.get("tool_calls"):
            print("Model decided to call functions:")
            for tool_call in message["tool_calls"]:
                print(f"Function name: {tool_call['function']['name']}")
                print(f"Arguments: {tool_call['function']['arguments']}")
        else:
            print("Reply:", message["content"])
    else:
        print("Error:", response.status_code, response.text)

2. Streaming Function Calling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import requests
    import json

    url = "http://localhost:8088/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Mathematical expression"}
                    },
                    "required": ["expression"]
                }
            }
        }
    ] 