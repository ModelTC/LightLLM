.. _openai_responses_api:

OpenAI Responses API (Experimental)
===================================

LightLLM can expose a ``/v1/responses`` endpoint that speaks the OpenAI
Responses API wire protocol. This is useful if you have client code written
against the newer OpenAI Python/TypeScript SDK (``client.responses.create``),
Codex CLI, or the OpenAI Agents SDK and want to point it at a locally hosted
open-source model without rewriting the client.

Enabling
--------

The ``/v1/responses`` endpoint is always exposed; no extra flag and no extra
Python dependency are required:

.. code-block:: bash

    python -m lightllm.server.api_server \
        --model_dir /path/to/model \
        --port 8088

Using it from the OpenAI SDK
----------------------------

.. code-block:: python

    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8088/v1",
        api_key="dummy",
    )
    resp = client.responses.create(
        model="any-name",  # echoed back; LightLLM serves the loaded model
        input="hello",
        max_output_tokens=1024,
    )
    print(resp.output_text)

Streaming works the same way the OpenAI SDK expects:

.. code-block:: python

    with client.responses.stream(
        model="any-name",
        input="Count from 1 to 5.",
        max_output_tokens=256,
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)

Function tools round-trip:

.. code-block:: python

    resp = client.responses.create(
        model="any-name",
        input="What's the weather in Paris?",
        tools=[{
            "type": "function",
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }],
    )
    for item in resp.output:
        if item.type == "function_call":
            print(item.name, item.arguments)

Supported features
------------------

- Text generation (streaming and non-streaming)
- ``instructions`` as a top-level system prompt
- Role-based ``input`` items with ``input_text`` / ``output_text`` content parts
- Function tools (definitions, tool calls, and ``function_call_output``
  follow-up turns)
- ``tool_choice`` (``auto`` / ``none`` / ``required`` / ``{type:"function", name:...}``)
- Vision inputs via ``input_image`` content parts
- Usage reporting (``input_tokens`` / ``output_tokens`` / ``total_tokens``)

Known limitations
-----------------

- Built-in tools (``web_search``, ``file_search``, ``code_interpreter``,
  ``computer_use_preview``) are not supported and are silently dropped from
  the request.
- Stateful fields (``store``, ``previous_response_id``) are accepted but
  ignored; LightLLM is stateless and does not persist responses.
- Structured outputs via ``text.format.json_schema`` are accepted but the
  ``text`` field is ignored — use the equivalent Chat Completions
  ``response_format`` via the ``/v1/chat/completions`` endpoint if you need
  schema-constrained generation.
- ``reasoning.summary`` is passed through from the underlying model's
  ``reasoning_content`` when present, but LightLLM does not synthesise
  summaries on its own.
- Model name is accepted but ignored; LightLLM always serves the model
  loaded via ``--model_dir`` and echoes the requested name back in the
  response.
