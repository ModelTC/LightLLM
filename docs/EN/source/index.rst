Welcome to Lightllm!
====================

.. figure:: ./assets/logos/lightllm-logo.png
  :width: 100%
  :align: center
  :alt: Lightllm
  :class: no-scaled-link

.. raw:: html

   <p style="text-align:center">
   <strong>A Lightweight and High-Performance Large Language Model Service Framework
   </strong>
   </p>

   <p style="text-align:center">
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/ModelTC/lightllm" data-show-count="true" data-size="large" aria-label="Star">Star</a>
   <a class="github-button" href="https://github.com/ModelTC/lightllm/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
   <a class="github-button" href="https://github.com/ModelTC/lightllm/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
   </p>


Lightllm is a pure Python-based large language model inference and serving framework, featuring lightweight design, easy extensibility, and high performance.
Lightllm integrates the advantages of numerous open-source solutions, including but not limited to FasterTransformer, TGI, vLLM, SGLang, and FlashAttention.

**Key Features**:

* Multi-process Collaboration: Input text encoding, language model inference, visual model inference, and output decoding are performed asynchronously, significantly improving GPU utilization.
* Cross-process Request Object Sharing: Through shared memory, cross-process request object sharing is achieved, reducing inter-process communication latency.
* Efficient Scheduling Strategy: Peak memory scheduling strategy with prediction, maximizing GPU memory utilization while reducing request eviction.
* High-performance Inference Backend: Efficient operator implementation, support for multiple parallelization methods (tensor parallelism, data parallelism, and expert parallelism), dynamic KV cache, rich quantization support (int8, fp8, int4), structured output, and multi-result prediction.

Documentation List
------------------

.. toctree::
   :maxdepth: 1
   :caption: Quick Start

   Installation Guide <getting_started/installation>
   Quick Start <getting_started/quickstart>
   Performance Benchmark <getting_started/benchmark>

.. toctree::
   :maxdepth: 1
   :caption: Deployment Tutorials

   DeepSeek R1 Deployment <tutorial/deepseek_deployment>
   Multi-Level Cache Deployment <tutorial/multi_level_cache_deployment>
   Multimodal Deployment <tutorial/multimodal>
   Reward Model Deployment <tutorial/reward_model>
   OpenAI api Usage <tutorial/openai>
   APIServer Parameters <tutorial/api_server_args_zh>
   Lightllm API Introduction <tutorial/api_param>
   
.. toctree::
   :maxdepth: 1
   :caption: Model Support

   Supported Models List <models/supported_models>
   Adding New Models <models/add_new_model>
   
.. toctree::
   :maxdepth: 1
   :caption: Architecture Introduction

   Architecture Overview <framework/framework>
   Token Attention <framework/token_attention>
   Efficient Router <framework/router>
   
.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
