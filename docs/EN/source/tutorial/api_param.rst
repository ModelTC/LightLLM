API Call Details
================

:code:`GET /health`
~~~~~~~~~~~~~~~~~~~
:code:`HEAD /health`
~~~~~~~~~~~~~~~~~~~~
:code:`GET /healthz`
~~~~~~~~~~~~~~~~~~~~

Get the current server running status

**Call Example**: 

.. code-block:: console

    $ curl http://0.0.0.0:8080/health

**Output Example**:

.. code-block:: python

    {"message":"Ok"}

:code:`GET /token_load`
~~~~~~~~~~~~~~~~~~~~~~~

Get the current server token usage status

**Call Example**: 

.. code-block:: console

    $ curl http://0.0.0.0:8080/token_load

**Output Example**:

.. code-block:: python

    {"current_load":0.0,"logical_max_load":0.0,"dynamic_max_load":0.0}

:code:`POST /generate`
~~~~~~~~~~~~~~~~~~~~~~

Call the model to implement text completion

**Call Example**: 

.. code-block:: console

    $ curl http://localhost:8080/generate \
    $ -H "Content-Type: application/json" \
    $ -d '{
    $      "inputs": "What is AI?",
    $      "parameters":{
    $        "max_new_tokens":17,
    $        "frequency_penalty":1
    $      },
    $      "multimodal_params":{}
    $     }'

**Output Example**:

.. code-block:: python

    {"generated_text": [" What is the difference between AI and ML? What are the differences between AI and ML"], "count_output_tokens": 17, "finish_reason": "length", "prompt_tokens": 4}

:code:`POST /generate_stream`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stream return text completion results

**Call Example**: 

.. code-block:: console

    $ curl http://localhost:8080/generate_stream \
    $ -H "Content-Type: application/json" \
    $ -d '{
    $      "inputs": "What is AI?",
    $      "parameters":{
    $        "max_new_tokens":17,
    $        "frequency_penalty":1
    $      },
    $      "multimodal_params":{}
    $     }'

**Output Example**:

::

    data:{"token": {"id": 3555, "text": " What", "logprob": -1.8383026123046875, "special": false, "count_output_tokens": 1, "prompt_tokens": 4}, "generated_text": null, "finished": false, "finish_reason": null, "details": null}

    data:{"token": {"id": 374, "text": " is", "logprob": -0.59185391664505, "special": false, "count_output_tokens": 2, "prompt_tokens": 4}, "generated_text": null, "finished": false, "finish_reason": null, "details": null}

    data:{"token": {"id": 279, "text": " the", "logprob": -1.5594439506530762, "special": false, "count_output_tokens": 3, "prompt_tokens": 4}, "generated_text": null, "finished": true, "finish_reason": "length", "details": null}

Stop conditions
~~~~~~~~~~~~~~~

For ``/generate`` and ``/generate_stream``, set ``parameters.stop_sequences`` to a
string or a list of strings. ``parameters.include_stop_str_in_output`` defaults to
``false``: the matched stop string and any following text in the same token are
removed. Set it to ``true`` to retain the stop string. This option applies to
string stops only.

Strings retain their original text and are never encoded into token ID sequences.
``DETOKEN_ALL`` matches decoded text during detokenization, with the HTTP output
layer handling buffering and trimming. ``OUTPUT_ALL`` and ``OUTPUT_CONTENT_ONLY``
match strings in the HTTP output layer.

Explicit token ID sequences are also supported, for example
``"stop_sequences": [[100, 101]]``; replace these IDs with the values used by your
model. These stops match independently in inference and HTTP output, including reasoning,
regardless of ``str_stop_match_mode`` or ``LIGHTLLM_STOP_IN_REASONING``.
HTTP output matches token ID stops before string stops. Each choice retains a bounded
tail of actual generated IDs across P/D capacity segments, including special tokens
whose decoded text is empty. A match preserves token text and metadata and changes
only the finish status to stop. IDs from the user prompt do not participate in matching.
``include_stop_str_in_output`` does not change token ID stop output behavior.
Strings and token ID sequences may be mixed, for example ``["END", [100, 101]]``.
RL, routed-expert output, and prompt-logprob requests bypass both HTTP stop wrappers
and rely on backend matching. HTTP token ID matching across P/D segments is disabled
for these requests.

.. list-table:: Stop matching scope
   :header-rows: 1

   * - ``str_stop_match_mode``
     - String stops
     - Explicit token ID stops
   * - ``0`` / ``DETOKEN_ALL`` (default)
     - All generated text
     - All generated tokens
   * - ``1`` / ``OUTPUT_ALL``
     - All generated text
     - All generated tokens
   * - ``2`` / ``OUTPUT_CONTENT_ONLY``
     - Answer content only, excluding reasoning
     - All generated tokens

Both streaming and non-streaming output follow this setting. When the output
ends with a prefix of any stop string, streaming holds the complete tokens
covering that prefix until the match is resolved. This prevents a stop split
across tokens from leaking into output.
An unmatched tail is returned when generation finishes. Token IDs, logprobs,
output token counts, and the finish reason retain their original meanings.

``/generate`` and ``/generate_stream`` match string stops against all generated text by default,
including reasoning text, even when a reasoning parser is configured.
For reasoning / content stop scope in OpenAI ``/v1/chat/completions``, see the
:ref:`OpenAI API examples <openai_api>`.

:code:`POST /get_score`
~~~~~~~~~~~~~~~~~~~~~~~
Reward model, get conversation score

**Call Example**: 

.. code-block:: python

    import json
    import requests

    query = "<|im_start|>user\nHello! What's your name?<|im_end|>\n<|im_start|>assistant\nMy name is InternLM2! A helpful AI assistant. What can I do for you?<|im_end|>\n<|reward|>"

    url = "http://127.0.0.1:8080/get_score"
    headers = {'Content-Type': 'application/json'}

    data = {
        "chat": query,
        "parameters": {
            "frequency_penalty":1
        }
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        print(f"Result: {response.json()}")
    else:
        print(f"Error: {response.status_code}, {response.text}")

**Output Example**:

::

    Result: {'score': 0.4892578125, 'prompt_tokens': 39, 'finish_reason': 'stop'} 