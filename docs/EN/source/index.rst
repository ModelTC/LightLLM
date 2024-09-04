Welcome Lightllm!
==================

.. figure:: ./assets/logos/lightllm-logo.png
  :width: 100%
  :align: center
  :alt: Lightllm
  :class: no-scaled-link

.. raw:: html

   <p style="text-align:center">
   <strong>A Light and Fast inference Services for LLM
   </strong>
   </p>

   <p style="text-align:center">
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/ModelTC/lightllm" data-show-count="true" data-size="large" aria-label="Star">Star</a>
   <a class="github-button" href="https://github.com/ModelTC/lightllm/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
   <a class="github-button" href="https://github.com/ModelTC/lightllm/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
   </p>


LightLLM is a Python-based LLM (Large Language Model) inference and serving framework, notable for its lightweight design, easy scalability, and high-speed performance. LightLLM harnesses the strengths of numerous well-regarded open-source implementations, including but not limited to FasterTransformer, TGI, vLLM, and FlashAttention.

**Features**:

* Tri-process asynchronous collaboration: tokenization, model inference, and detokenization are performed asynchronously, leading to a considerable improvement in GPU utilization.
* Nopad (Unpad): offers support for nopad attention operations across multiple models to efficiently handle requests with large length disparities.
* Dynamic Batch: enables dynamic batch scheduling of requests
* FlashAttention: incorporates FlashAttention to improve speed and reduce GPU memory footprint during inference.
* Tensor Parallelism: utilizes tensor parallelism over multiple GPUs for faster inference.
* Token Attention: implements token-wise's KV cache memory management mechanism, allowing for zero memory waste during inference.
* High-performance Router: collaborates with Token Attention to meticulously manage the GPU memory of each token, thereby optimizing system throughput.
* Int8KV Cache: This feature will increase the capacity of tokens to almost twice as much. only llama support.

**Supported Model List**：

- `BLOOM <https://huggingface.co/bigscience/bloom>`_
- `LLaMA <https://github.com/facebookresearch/llama>`_
- `LLaMA V2 <https://huggingface.co/meta-llama>`_
- `StarCoder <https://github.com/bigcode-project/starcoder>`_
- `Qwen-7b <https://github.com/QwenLM/Qwen-7B>`_
- `ChatGLM2-6b <https://github.com/THUDM/ChatGLM2-6B>`_
- `Baichuan-7b <https://github.com/baichuan-inc/Baichuan-7B>`_
- `Baichuan2-7b <https://github.com/baichuan-inc/Baichuan2>`_
- `Baichuan2-13b <https://github.com/baichuan-inc/Baichuan2>`_
- `Baichuan-13b <https://github.com/baichuan-inc/Baichuan-13B>`_
- `InternLM-7b <https://github.com/InternLM/InternLM>`_
- `Yi-34b <https://huggingface.co/01-ai/Yi-34B>`_
- `Qwen-VL <https://huggingface.co/Qwen/Qwen-VL>`_
- `Qwen-VL-Chat <https://huggingface.co/Qwen/Qwen-VL-Chat>`_
- `Llava-7b <https://huggingface.co/liuhaotian/llava-v1.5-7b>`_
- `Llava-13b <https://huggingface.co/liuhaotian/llava-v1.5-13b>`_
- `Mixtral <https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1>`_
- `Stablelm <https://huggingface.co/stabilityai/stablelm-2-1_6b>`_
- `MiniCPM <https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16>`_
- `Phi-3 <https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3>`_
- `CohereForAI <https://huggingface.co/CohereForAI/c4ai-command-r-plus>`_
- `DeepSeek-V2-Lite <https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite>`_ 
- `DeepSeek-V2 <https://huggingface.co/deepseek-ai/DeepSeek-V2>`_ 


Docs List
-------------

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 1
   :caption: Lightllm

   lightllm/lightllm_intro
   lightllm/lightllm_impl

.. toctree::
   :maxdepth: 1
   :caption: Model

   Supported Model <models/supported_models>
   Examples <models/test>
   Add new models <models/add_new_model>

.. toctree::
   :maxdepth: 1
   :caption: Launching Server

   Server Args <server/api_server_args>
   Benchmark <server/benchmark>


.. toctree::
   :maxdepth: 1
   :caption: Using Server

   user/api_param
   user/openapi_docs
   
   
.. toctree::
   :maxdepth: 1
   :caption: development docs

   dev/token_attention
   dev/router
   
.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
