欢迎了解 Lightllm!
==================

.. figure:: ./assets/logos/lightllm-logo.png
  :width: 100%
  :align: center
  :alt: Lightllm
  :class: no-scaled-link

.. raw:: html

   <p style="text-align:center">
   <strong>一个轻量级、高性能的大语言模型服务框架
   </strong>
   </p>

   <p style="text-align:center">
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/ModelTC/lightllm" data-show-count="true" data-size="large" aria-label="Star">Star</a>
   <a class="github-button" href="https://github.com/ModelTC/lightllm/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
   <a class="github-button" href="https://github.com/ModelTC/lightllm/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
   </p>


Lightllm 是一个纯python开发的大语言模型推理和服务框架，具有轻量级设计、易扩展以及高性能等特点。
Lightllm 整合了众多的开源方案的优点，包括但不限于 FasterTransformer、TGI、vLLM 和 FlashAttention。


**重要特性**:

* 多进程协同：分词、语言模型推理、视觉模型推理、分词等工作异步进行，大幅提高GPU利用率。
* 零填充：提供对跨多个模型的 nopad-Attention 计算的支持，以有效处理长度差异较大的请求。
* 动态批处理：能够对请求进行动态的批处理调度。
* FlashAttention：结合 FlashAttention 来提高推理过程中的速度并减少 GPU 内存占用。
* 向量并行：利用多个 GPU 进行张量并行性从而加快推理速度。
* **Token Attention**：实现了以token为单位的KV缓存内存管理机制，实现推理过程中内存零浪费。
* 高性能路由：结合Token Attention，对GPU内存以token为单位进行精致管理，优化系统吞吐量。
* int8 KV Cache：该功能可以将最大token量提升解决两倍。现在只支持llama架构的模型。

**支持的模型列表**：

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


文档列表
-------------

.. toctree::
   :maxdepth: 1
   :caption: 快速入门

   安装指南 <getting_started/installation>
   快速开始 <getting_started/quickstart>


.. toctree::
   :maxdepth: 1
   :caption: lightllm原理

   lightllm介绍 <lightllm_info/lightllm>


.. toctree::
   :maxdepth: 1
   :caption: 模型

   支持的模型列表 <models/supported_models>
   启动和测试模型示例 <models/test>
   添加新模型 <models/add_new_model>

.. toctree::
   :maxdepth: 1
   :caption: 启动服务

   启动参数说明 <server/api_server_args_zh>
   服务性能评测 <server/benchmark>


.. toctree::
   :maxdepth: 1
   :caption: 使用服务

   user/api_param
   user/openapi_docs
   user/param_class/index
   
   
.. toctree::
   :maxdepth: 1
   :caption: 开发者文档

   dev/lightllm_impl
   dev/token_attention
   dev/router
   
.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
