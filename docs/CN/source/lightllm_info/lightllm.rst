.. _lightllm:

LightLLM介绍
================

随着ChatGPT的流行，大语言模型(简称LLM)受到越来越多的关注。此类模式的出现，极大地提高了人们的工作效率。
然而，进一步广泛使用LLM的关键在于如何以低成本和高吞吐量地部署数十亿参数的模型。
为了提高大模型服务的吞吐量，让更多感兴趣的研究人员快速参与进来，
一种名为 LightLLM 的轻量级 LLM 推理服务框架应运而生。 
LightLLM 引入了一种更细粒度的kvCache管理算法，称为TokenAttention，
并设计了一个与TokenAttention高效配合的Efficient Router调度算法。
通过 TokenAttention 和 Efficient Router 的配合，
LightLLM 在大多数场景下实现了比 vLLM 和 Text Generation Inference 更高的吞吐量，
甚至在某些情况下性能提升了 4 倍左右。 LightLLM 灵活、用户友好且高效，
欢迎有兴趣的朋友进入 `项目主页 <https://github.com/ModelTC/lightllm>`_ 了解更多。


.. _challenge:

LLM 推理服务的挑战
------------------

大型语言模型由于其优异的性能而引起了研究人员的极大关注。
这些模型不仅可以与人类进行日常对话，还可以帮助完成各种日常任务，从而提高生产力。
然而，尽管这些模型表现出了出色的性能，但提高部署大规模模型的性能仍面临以下挑战：

* **内存碎片严重**：从几十到几百G不等的网络权重，以及推理过程中不断动态增长的KV Cache，容易导致大量的内存碎片，进而导致内存利用率低。
* **请求调度效率低**：请求的长度随时间动态变化，可能导致GPU空闲或利用率低的问题。
* **内核定制难度大**：为了高效利用内存、提高服务吞吐量，需要针对网络定制内核。然而，这需要研究人员付出大量的努力。


.. _solutions_and_problems:

现有的解决方案和存在问题
-----------------------------

为了应对上述挑战，许多优秀的LLM推理框架应运而生，
例如FasterTransformer、Text-Generation-Inference（简称TGI）、vLLM等。这些框架的核心特性和能力如下表所示：


.. list-table:: 各个框架对比
   :header-rows: 1

   * - 框架
     - NV Triton + FasterTransformer
     - TGI
     - vLLM
     - LightLLM
   * - 核心特征
     - 高效算子
     - `Continuous batch <https://github.com/huggingface/text-generation-inference/tree/main/router>`_, Token streaming
     - `PageAttention <https://vllm.ai/>`_
     - 三进程异步协同, `Token Attention <https://github.com/ModelTC/lightllm/blob/main/docs/TokenAttention.md>`_, Efficient Router
   * - 内存碎片
     - 少
     - 多
     - 少
     - 少
   * - 请求的调度效率
     - 低
     - 中
     - 中
     - 高
   * - 定制化算子的难度
     - 高
     - 中
     - 中
     - 低

这些框架都有自己独特的特点。
例如，FasterTransformer具有优异的静态推理性能，但缺乏健壮的服务调度，并且主要采用C++开发，导致二次开发成本较高。 
TGI具有优秀的服务接口和Continuation Batch等调度特性，但其推理性能、调度策略、内存管理等方面存在一些不足。 
vLLM具有出色的内存管理能力，但在请求调度方面缺乏效率，其整体实现细节更适合部署小型模型。


Lightllm
----------------------

因此，为了解决这些问题，我们开发了一个名为LightLLM的LLM部署框架，它是基于纯Python语言的。
它使研究人员能够在本地轻松部署和定制轻量级模型，从而可以快速扩展不同模型并集成各种优秀的开源功能。 
LightLLM的核心特点如下：

* 三进程异步协作：分词、模型推理、去分词异步进行，GPU利用率大幅提升。
* :ref:`TokenAttention`：实现token-wise的KV缓存内存管理机制，实现推理时内存零浪费。
* :ref:`Efficient_Router`：与Token Attention合作，精心管理每个Token的GPU内存，从而优化系统吞吐量。

凭借基于OpenAI Triton开发的高度协调的高效内核和服务调度，LightLLM实现了优异的吞吐性能。

.. figure:: ../assets/lightllm/arch.png
  :width: 100%
  :align: center
  :alt: Lightllm
  :class: no-scaled-link



性能评测
-----------

我们使用当前主流推理框架TGI、NV Triton + FasterTransformer、vLLM在ShareGPT_Vicuna_unfiltered数据集上进行了性能比较。结果如下图所示。可以看出，LightLLM 在不同模型大小上实现了更高的吞吐量。 TGI内存碎片严重，难以实现高吞吐量。 vLLM引入了PageAttention，但由于其整体实现细节更利于小模型推理，因此在大模型上的并发性能不是很理想（使用默认配置）。相比之下，LightLLM 在各种模型尺寸上都保持了稳健的性能，并且在大型模型 (LLaMA-65B) 上比 TGI 和 vLLM 提高了约 2-3 倍。

.. image:: ../assets/lightllm/Performance.png
   :alt: Efficient_Router1
   :align: center


TGI兼容性和消融分析为了进一步验证TokenAttention和Router的有效性，我们还将这些功能集成到TGI中以解决其内存碎片问题，如下图（左）所示。可以看出，引入TokenAttention和Router后，与原始TGI相比，性能提升了4倍以上。

长短混合请求情况下的改进：从下图（左）可以看出，Router的引入并没有带来更明显的性能提升，这是由于问题长度的差异ShareGPT_Vicuna_unfiltered 的数据集并不重要。为此，我们构建了长度差异较大的请求集合，并验证了高效路由器的性能。结果如下所示（右）。可以看到，我们的Efficient Router可以更好地利用GPU资源，对于问题长度差异较大的请求可以带来近50%的性能提升。


.. image:: ../assets/lightllm/Performance2.png
   :alt: Efficient_Router1
   :align: center


左图展示了LightLLM和TGI的兼容性以及消融分析，右图展示了我们的Efficient Router对长短请求的增强


未来工作
---------

* 支持更多的模型
* 增强路由调度算法
* 高性能的 int8 和 int4 仅权重的 kv cache 的支持
* 全量化模型的支持
* 混合精度模型
* 稀疏化

LightLLM致力于让更多人参与进来，从而灵活高效地探索各种LLM部署和推理解决方案。也为硬件厂商推动该领域的发展提供参考。我们希望大家能够给它更多的star，fork这个项目，并做出贡献。我们相信未来将会出现更多的技术和解决方案（如TensorRT），不断降低部署成本，让AGI更容易走进普通家庭。
