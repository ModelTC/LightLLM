.. _Efficient_Router:

Efficient Router
===================

引入高效路由器来管理传入请求，并动态确定该请求是否可以与已运行的推理批次融合。
合并标准是估计合并推理过程中最大Token占用量是否小于硬件可容纳的最大容量。
这里，我们将这个最大容量设置为 ``max_total_token_num``。在 **Token Attention** 的支持下，我们可以准确地管理Token的使用情况，并且可以确保永远不会出现内存不足（out-of-memory）的情况。

.. image:: ../assets/lightllm/ER1.png
   :alt: Efficient_Router1
   :align: center


如上图所示，每一行代表一个请求当前的运行状态，黄色代表已经运行过的历史kv缓存token，每个格子代表一个token，灰色代表要生成的token。
生成的Token数量由每个请求设置的最大输出长度和已生成的Token数量决定。
上图中，绿色网格的第二行表示新到达的请求，图中按照要生成的输出的长度升序列出了所有请求。

如果我们假设新的请求融合成一个Batch进行推理，那么最大的token使用量必然会出现在时间点1、时间2、时间3中的一个时间点，我们只需要计算这些时间点的token使用量是否达到最大值即可。三个时间点都没有超过max_total_token_num，说明新的请求可以加入到Batch中进行融合推理。

时间1的总使用代币等于黄色单元格数量加上绿色单元格数量（见下图）

.. image:: ../assets/lightllm/ER2.png
   :alt: Efficient_Router1
   :align: center


时间2的总使用代币等于黄色方块的数量加上绿色方块的数量（见下图）

.. image:: ../assets/lightllm/ER3.png
   :alt: Efficient_Router1
   :align: center

时间3的总使用代币等于黄色方块的数量（见下图）

.. image:: ../assets/lightllm/ER4.png
   :alt: Efficient_Router1
   :align: center

实际最大令牌使用量始终为时间 1、时间 2 或时间 3 之一。

只要动态推理过程中token的最大使用量低于max_total_token_num，就说明可以批量进行新的请求进行推理。

为了快速计算批次中所有请求所需的最大令牌使用量，我们使用 numpy 实现了一个高效的示例。


.. code-block:: python

    import numpy as np

    def demo():
        max_total_token_num = 100
        req_list = [(5, 4), (4, 3), (5, 3), (3, 2), (4, 2)]  # (run_len, left_output_len)
        req_list.sort(key=lambda x: -x[1])

        left_out_len_array = np.array([e[1] for e in req_list])
        has_run_len_array = np.array([e[0] for e in req_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(req_list) + 1, 1)
        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()

        if need_max_token_num <= max_total_token_num:
            print("ok")
        else:
            print("oom")