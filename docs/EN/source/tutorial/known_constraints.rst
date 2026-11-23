Known Usage Constraints
=======================

This page records accepted usage constraints of the current implementation.
They define interface support boundaries rather than tracked bugs. Callers of
the relevant low-level interfaces are responsible for observing these constraints.

Custom reasoning-end markers during PD continuation
------------------------------------------------------------

Scope
^^^^^

This constraint applies to requests that supply
``parameters.guided_reasoning_end`` through ``/generate``, ``/generate_stream``,
or TGI-compatible interfaces that forward sampling parameters, and actually
undergo PD continuation across segments. It concerns compatibility between custom
markers and re-encoding historical text, for both text and multimodal requests.

``/v1/chat/completions`` derives the marker from the reasoning parser, tokenizer,
and thinking mode on the server. It does not expose this custom parameter;
standard Chat callers do not need to supply reasoning-end token IDs.

Why the constraint exists
^^^^^^^^^^^^^^^^^^^^^^^^^

The PD Master stores historical output as text. To continue a request, it appends
that text to the original prompt, re-encodes it, and replays the historical suffix
to restore the thinking budget and answer constraint state. Re-encoding the same
text does not guarantee the original token sequence. A custom marker made from
ordinary tokens can change even when the token count stays the same.

For example, in the verified Qwen3.5-0.8B tokenizer, ``151668`` represents ordinary
text ``خبار``, while the standard ``</think>`` token is ``248069``. A historical
text fragment was re-encoded as follows:

.. code-block:: text

   Original tokens:   [220, 151668]
   Re-encoded tokens: [71787, 148798]

With a budget of 7 and marker ``[151668]``, replay expects ``151668`` at the
budget boundary but encounters ``148798``, recording
``Grammar rejected sampled token 148798``. This is a constraint rejection during
history restoration. ``148798`` is still a valid vocabulary ID, so vocabulary
range validation cannot prevent this case.

Failure behavior and usage requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* An incompatible request may end early with ``finish_reason="error"`` and must
  not be treated as successful. The request's constraint state catches and
  isolates this error; the service process can continue handling other requests.
* Encode the parser's standard reasoning-end sequence with the current model's
  tokenizer. Do not reuse hardcoded token IDs across models.
* Arbitrary ordinary-token or multi-token custom sequences are not guaranteed
  to restore correctly after PD text re-encoding. Callers who need custom markers
  should verify their stability with the target tokenizer and historical context,
  including actual continuation across segments. A passing single-segment request
  is insufficient.

The native interfaces retain custom markers without promising PD continuation
compatibility for every custom sequence.
