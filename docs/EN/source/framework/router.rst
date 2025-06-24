.. _Efficient_Router:

Efficient Router
================

Introducing an efficient router to manage incoming requests and dynamically determine whether the request can be merged with already running inference batches.
The merge criterion is whether the estimated maximum token usage during merged inference is less than the maximum capacity that the hardware can accommodate.
Here, we set this maximum capacity to ``max_total_token_num``. With the support of **Token Attention**, we can accurately manage token usage and ensure that out-of-memory situations never occur.

.. image:: ../assets/lightllm/ER1.png
   :alt: Efficient_Router1
   :align: center

As shown in the figure above, each row represents the current running state of a request, yellow represents historical kv cache tokens that have already been run, each cell represents a token, and gray represents tokens to be generated.
The number of generated tokens is determined by the maximum output length set for each request and the number of tokens already generated.
In the figure above, the second row of green grid represents a newly arrived request, and all requests are listed in ascending order according to the length of output to be generated.

If we assume that the new request is merged into a batch for inference, the maximum token usage will necessarily appear at one of time point 1, time 2, or time 3. We only need to calculate whether the token usage at these time points reaches the maximum value. If none of the three time points exceed max_total_token_num, it means the new request can be added to the batch for merged inference.

Total token usage at time 1 equals the number of yellow cells plus the number of green cells (see figure below)

.. image:: ../assets/lightllm/ER2.png
   :alt: Efficient_Router1
   :align: center

Total token usage at time 2 equals the number of yellow squares plus the number of green squares (see figure below)

.. image:: ../assets/lightllm/ER3.png
   :alt: Efficient_Router1
   :align: center

Total token usage at time 3 equals the number of yellow squares (see figure below)

.. image:: ../assets/lightllm/ER4.png
   :alt: Efficient_Router1
   :align: center

The actual maximum token usage is always one of time 1, time 2, or time 3.

As long as the maximum token usage during dynamic inference is below max_total_token_num, it means new requests can be batched for inference.

To quickly calculate the maximum token usage required by all requests in the batch, we implemented an efficient example using numpy.

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