.. _installation:

安装指南
============

Lightllm 是一个纯python开发的推理框架，其中的算子使用triton编写。

环境要求
------------

* 操作系统: Linux
* Python: 3.9
* GPU: 计算能力 7.0 以上 (e.g., V100, T4, RTX20xx, A100, L4, H100, 等等.)

.. _build_from_docker:

使用docker安装
----------------
安装lightllm最简单的方法是使用官方镜像，你可以直接拉取官方镜像并运行：

.. code-block:: console

    $ # 拉取官方镜像
    $ docker pull ghcr.io/modeltc/lightllm:main
    $
    $ # 运行
    $ docker run -it --gpus all -p 8080:8080            \
    $   --shm-size 1g -v your_local_path:/data/         \
    $   ghcr.io/modeltc/lightllm:main /bin/bash

你也可以使用源码手动构建镜像并运行：

.. code-block:: console

    $ # 手动构建镜像
    $ docker build -t <image_name> .
    $
    $ # 运行
    $ docker run -it --gpus all -p 8080:8080            \
    $   --shm-size 1g -v your_local_path:/data/         \
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
    $ conda create -n lightllm python=3.9 -y
    $ conda activate lightllm
    $
    $ # 下载lightllm的最新源码
    $ git clone https://github.com/ModelTC/lightllm.git
    $ cd lightllm
    $
    $ # 安装lightllm的依赖 (cuda 12.4)
    $ pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
    $
    $ # 安装lightllm
    $ python setup.py install

NOTE: 如果您出于一些原因使用了cuda 11.x的torch, 请运行 `pip install nvidia-nccl-cu12==2.20.5` 以支持 torch cuda graph.

.. note::

    Lightllm 的代码在多种GPU上都进行了测试，包括 V100, A100, A800, 4090, 和 H800。
    如果你使用 A100 、A800 等显卡，那么推荐你安装 triton==3.0.0 ：

    .. code-block:: console

        $ pip install triton==3.0.0 --no-deps

    如果你使用 H800、V100 等显卡，那么推荐你安装 triton-nightly：

    .. code-block:: console

        $ pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly --no-deps
    
    具体原因可以参考：`issue <https://github.com/triton-lang/triton/issues/3619>`_ 和 `fix PR <https://github.com/triton-lang/triton/pull/3638>`_

