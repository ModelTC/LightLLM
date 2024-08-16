.. _lightllm:

LightLLM Overview
===========================

With the popularity of ChatGPT, large language model, abbreviated as LLM, has received increasing attention. The emergence of such models has greatly improved people's work efficiency. However, the key to further widespread adoption lies in how to deploy models with billons of parameters at low cost and high throughput. To improve the throughput of large model services and enable more interested researchers to quickly get involved, a lightweight LLM inference service framework called LightLLM has emerged. LightLLM introduces a more fine-grained kv cache management algorithm called TokenAttention and designs an Efficient Router scheduling implementation that works efficiently with TokenAttention. Through the interaction of TokenAttention and Efficient Router, LightLLM achieves higher throughput than vLLM and Text Generation Inference in most scenarios, with performance improvements of around 4 times in some cases. LightLLM is flexible, user-friendly, and efficient. Interested friends may want to click on the link below to try it out.

Project：https://github.com/ModelTC/lightllm

.. _challenge:

The challenge of LLM Serving
-------------------------------

Large language models have garnered significant attention from researchers due to their excellent performance. These models not only engage in everyday conversations with humans but also assist in completing various daily tasks, thereby enhancing productivity. However, despite the remarkable performance demonstrated by these models, deploying large-scale models to improve service performance poses the following challenges:

* **Severe fragmentation of memory**: Network weights ranging from tens to hundreds of gigabytes, as well as the constantly dynamic growing KV Cache during inference, easily leads to low memory utilization.
* **Low efficiency in request scheduling**: The length of requests dynamically changes over time, which can result in GPU idling or low utilization issues.
* **High difficulty in kernel customization**: Customizing kernels for networks is necessary to efficiently utilize memory and improve service throughput. However, it will require a significant amount of effort from researchers.

.. _solutions_and_problems:

Existing solutions and problems
-------------------------------------

To address the aforementioned challenges, many excellent LLM inference frameworks have emerged, such as FasterTransformer, Text-Generation-Inference (referred to as TGI), vLLM, etc. The core features and capability matrices of these frameworks are shown in the table below:

.. list-table:: Comparison of various frameworks
   :header-rows: 1

   * - Framework
     - NV Triton + FasterTransformer
     - TGI
     - vLLM
     - LightLLM
   * - core feature
     - 	Efficient kernel
     - `Continuous batch <https://github.com/huggingface/text-generation-inference/tree/main/router>`_, Token streaming
     - `PageAttention <https://vllm.ai/>`_
     - Tri-process asynchronous collaboration，:ref:`TokenAttention`，:ref:`Efficient_Router`
   * - Memory fragmentation
     - low
     - high
     - low
     - low
   * - Request scheduling efficiency
     - low
     - middle
     - middle
     - high
   * - Difficulty of kernel customization
     - high
     - middle
     - middle
     - low

These frameworks all have their own unique features. For example, FasterTransformer has excellent static inference performance but lacks robust service scheduling and is primarily developed in C++, resulting in high secondary development costs. TGI has excellent service interfaces and scheduling features such as Continuous Batch, but its inference performance, scheduling strategy, and memory management have some shortcomings. vLLM has excellent memory management but lacks efficiency in request scheduling, and its overall implementation details are more suitable for deploying small models.

Lightllm
----------------------

Therefore, to address these issues, we have developed a LLM deployment framework called LightLLM, which is based on the pure Python language. It enables researchers to easily deploy and customize lightweight models locally, allowing for rapid expansion of different models and integration of various excellent open-source features. The core features of LightLLM are as follows:

* Tri-process asynchronous collaboration: tokenization, model inference, and detokenization are performed asynchronously, leading to a considerable improvement in GPU utilization.
* :ref:`TokenAttention`: implements token-wise's KV cache memory management mechanism, allowing for zero memory waste during inference.
* :ref:`Efficient_Router`: collaborates with Token Attention to meticulously manage the GPU memory of each token, thereby optimizing system throughput.

With the highly coordinated efficient kernels developed based on OpenAI Triton and service scheduling, LightLLM achieves excellent throughput performance

.. figure:: ../assets/lightllm/arch.png
  :width: 100%
  :align: center
  :alt: Lightllm
  :class: no-scaled-link



LightLLM is committed to enabling more people to participate, allowing flexible and efficient exploration of various LLM deployment and inference solutions. It also serves as a reference for hardware manufacturers to promote the development of the field. We hope that everyone can give it more stars, fork the project, and contribute. We believe that in the future, more technologies and solutions (such as TensorRT) will emerge, continuously reducing deployment costs and making AGI more accessible to ordinary households.