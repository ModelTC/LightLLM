.. _installation:

安装指南
============

Lightllm 是一个纯python开发的推理框架，其中的算子使用triton编写。

环境要求
------------

* 操作系统: Linux
* Python: 3.10
* GPU: 计算能力 7.0 以上 (e.g., V100, T4, RTX20xx, A100, L4, H100, 等等.)

.. _build_from_docker:

使用docker安装
----------------
安装lightllm最简单的方法是使用官方镜像，你可以直接拉取官方镜像并运行：

.. code-block:: console

    $ # 拉取官方镜像
    $ docker pull ghcr.io/modeltc/lightllm:main
    $
    $ # 运行服务, 注意现在的lightllm服务非常的依赖共享内存部分，在启动
    $ # 前请确保你的docker设置中已经分配了足够的共享内存，否则可能导致
    $ # 服务无法正常启动。
    $ # 1.如果是纯文本服务，建议分配2GB以上的共享内存, 如果你的内存充足，建议分配16GB以上的共享内存.
    $ # 2.如果是多模态服务，建议分配16GB以上的共享内存，具体可以根据实际情况进行调整.
    $ # 如果你没有足够的共享内存，可以尝试在启动服务的时候调低 --running_max_req_size 参数，这会降低
    $ # 服务的并发请求数量，但可以减少共享内存的占用。如果是多模态服务，也可以通过降低 --cache_capacity
    $ # 参数来减少共享内存的占用。
    $ docker run -it --gpus all -p 8080:8080            \
    $   --shm-size 2g -v your_local_path:/data/         \
    $   ghcr.io/modeltc/lightllm:main /bin/bash

你也可以使用源码手动构建镜像并运行,建议手动构建镜像,因为更新比较频繁：

.. code-block:: console

    $ # 进入代码仓库的根目录
    $ cd /lightllm
    $ # 手动构建镜像。
    $ docker build -t <image_name> -f ./docker/Dockerfile .
    $
    $ # 运行
    $ docker run -it --gpus all -p 8080:8080            \
    $   --shm-size 2g -v your_local_path:/data/         \
    $   <image_name> /bin/bash

或者你也可以直接使用脚本一键启动镜像并且运行：

.. code-block:: console

    $ # 查看脚本参数
    $ python tools/quick_launch_docker.py --help

.. note::
    如果你使用多卡，你也许需要提高上面的 –shm_size 的参数设置。

.. _build_from_source:

使用源码安装
----------------

你也可以使用源码安装Lightllm：

.. code-block:: console

    $ # (推荐) 创建一个新的 conda 环境
    $ conda create -n lightllm python=3.10 -y
    $ conda activate lightllm
    $
    $ # 下载lightllm的最新源码
    $ git clone https://github.com/ModelTC/lightllm.git
    $ cd lightllm
    $
    $ # 安装lightllm的依赖 (CUDA 13.0)
    $ pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
    $
    $ # 安装lightllm的依赖 (摩尔线程 GPU)
    $ ./generate_requirements_musa.sh
    $ pip install -r requirements-musa.txt
    $
    $ # 安装lightllm的依赖 (Intel XPU)
    $ pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
    $ ./generate_requirements_xpu.sh
    $ pip install -r requirements-xpu.txt
    $ pip install --no-deps xgrammar==0.2.8
    $
    $ # 安装lightllm
    $ python setup.py install

Intel XPU
^^^^^^^^^

在 Intel GPU 上使用 ``pip install --no-deps -e .`` 安装 Lightllm（``setup.py`` 依赖 CUDA 版的 ``triton``，
XPU 版 torch 自带 ``pytorch-triton-xpu``）。硬件平台根据安装的 torch 自动识别（默认 ``--hardware_platform auto``），
也可以用 ``--hardware_platform xpu`` 显式指定：

.. code-block:: console

    $ ZE_AFFINITY_MASK=0 python -m lightllm.server.api_server --model_dir /path/to/Qwen3-8B \
    $     --tp 1 --max_req_total_len 24576

* 使用 Triton 注意力后端和算子，分布式后端为 ``xccl``。
* 默认关闭图捕获，可设置 ``LIGHTLLM_XPU_ENABLE_GRAPH=1`` 尝试开启。
* 使用 ``ZE_AFFINITY_MASK`` 选择显卡。
* 暂不支持：量化模型、``--enable_ep_moe``、``--mtp_mode``、PD 分离、``--enable_cpu_cache``、
  ``--enable_dp_prompt_cache_fetch``、``--enable_torch_memory_saver``。

使用 uv 安装
----------------

也可以使用 `uv <https://docs.astral.sh/uv/>`_ 安装 Lightllm：

.. code-block:: console

    $ # 创建 Python 3.10 虚拟环境
    $ uv venv --python 3.10
    $ source .venv/bin/activate
    $
    $ # 安装 Lightllm 依赖 (CUDA 13.0)
    $ uv pip install -r requirements.txt --torch-backend=cu130
    $
    $ # 安装 Lightllm
    $ uv pip install -e .
