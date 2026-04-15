.. _anthropic_api:

Anthropic Messages API (Experimental)
=====================================

LightLLM can expose a ``/v1/messages`` endpoint that speaks the Anthropic
Messages API wire protocol. This is useful if you have client code written
against the Anthropic Python/TypeScript SDK and want to point it at a locally
hosted open-source model without rewriting the client.

Enabling
--------

Install the optional dependency:

.. code-block:: bash

    pip install 'lightllm[anthropic_api]'

Start the server with the flag:

.. code-block:: bash

    python -m lightllm.server.api_server \
        --model_dir /path/to/model \
        --enable_anthropic_api \
        --port 8088

Using it from the Anthropic SDK
-------------------------------

.. code-block:: python

    import anthropic

    client = anthropic.Anthropic(
        base_url="http://localhost:8088",
        api_key="dummy",
    )
    resp = client.messages.create(
        model="any-name",  # echoed back; LightLLM serves the loaded model
        max_tokens=1024,
        messages=[{"role": "user", "content": "hello"}],
    )
    print(resp.content[0].text)

Streaming works the same way the Anthropic SDK expects:

.. code-block:: python

    with client.messages.stream(
        model="any-name",
        max_tokens=256,
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

Supported features
------------------

- Text generation (streaming and non-streaming)
- System prompts
- Tool use / function calling
- Multi-turn conversations
- Vision (image inputs) via Anthropic content blocks

Known limitations
-----------------

- Prompt caching (``cache_control``) is accepted but ignored; ``cache_*``
  fields in ``usage`` are always zero.
- Extended thinking (``thinking`` parameter) is not supported.
- The Batch API (``/v1/messages/batches``) and Files API are not implemented.
- Model name is accepted but ignored; LightLLM always serves the model
  loaded via ``--model_dir`` and echoes the requested name back in the response.
- On the streaming path, ``message_start.message.usage.input_tokens`` is
  always ``0`` because the upstream usage chunk arrives after all content
  chunks. Clients that need an accurate prompt-token count should read
  ``message_delta.usage`` at the end of the stream.
