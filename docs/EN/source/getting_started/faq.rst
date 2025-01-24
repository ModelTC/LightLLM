.. _faq:

- The LLaMA tokenizer fails to load.
    - Consider resolving this by running the command:

      .. code-block:: shell

         pip install protobuf==3.20.0

- ``error   : PTX .version 7.4 does not support .target sm_89``
    - Launch with:

      .. code-block:: shell

         bash tools/resolve_ptx_version python -m lightllm.server.api_server ...